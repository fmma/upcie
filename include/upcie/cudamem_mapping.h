// SPDX-License-Identifier: BSD-3-Clause

/**
 * Registry of externally-allocated CUDA buffers for NVMe DMA
 * ==========================================================
 *
 * Where cudamem_heap pre-allocates a single dma-buf-backed range and serves
 * sub-allocations from it, cudamem_mapping registers buffers that the caller
 * already allocated (via cuMemAlloc, cuMemCreate/cuMemMap, or cudaMalloc) and
 * builds a physical address LUT for them via the same dma-buf interface.
 *
 * One dma-buf per registration
 * ----------------------------
 *
 * Each `_add` call exports one dma-buf for the registration's
 * `alloc_granularity`-floored range via a single
 * cuMemGetHandleForAddressRange. A registration whose floored range is fully
 * contained in one already in the registry reuses the existing dma-buf via a
 * refcounted handle struct, so registering the same buffer twice (or
 * registering a sub-range of an existing buffer) consumes one fd, not two.
 * Compared to a per-chunk dma-buf scheme, this collapses fd consumption from
 * N_chunks to N_handles for large contiguous registrations: an 8 GiB buffer
 * at 2 MiB granularity goes from 4096 fds to 1 fd.
 *
 * Registrations whose floored range partially overlaps an existing handle
 * without containment are rejected with -EEXIST. Splitting the existing
 * handle or holding two dma-bufs over shared chunks would both add
 * complexity without a known caller.
 *
 * The floored range may span multiple alloc_granularity-sized chunks; the
 * dma-buf LUT supplies phys for all of them in one CUDA round-trip. Each
 * chunk's BAR1 IOVA window is contiguous (BAR1 page-table large-page
 * guarantee), so each chunk has one `phys_base` (held in
 * `lut_phys[chunk_idx]`), and per-page phys is computed during virt_to_phys
 * as
 *
 *     phys = lut_phys[chunk_idx] + (vaddr & (alloc_granularity - 1))
 *
 * `_add` walks chunks intersecting the floored user range, ref-bumps existing
 * entries, and populates new ones from the handle's dma-buf LUT. `_remove`
 * decrements per-chunk rc and clears the lut_phys slot when rc reaches zero;
 * the handle's own rc tracks fd lifetime independently and the dma-buf is
 * detached when the last sharing registration goes away.
 *
 * Registrations and chunks form a many-to-many relationship: one registration
 * may span multiple chunks (large or chunk-boundary-crossing buffers), and
 * one chunk may be referenced by multiple registrations (sub-granularity
 * allocations packed by the driver, or a sub-range registration that reuses
 * a containing handle). The chunk's `rc` counts the registrations covering
 * it; the handle's `rc` counts the registrations sharing it (those whose
 * floored range is contained in the handle's range).
 *
 * Lookup table layout
 * -------------------
 *
 * Chunks are indexed directly by chunk_idx = vaddr >> alloc_granularity_shift,
 * over the full user virtual address space (CUDAMEM_MAPPING_VA_BITS, default
 * 48). Two parallel arrays cover the chunk_idx range:
 *
 *   lut_phys[chunk_idx] -> uint64_t phys_base (0 == unmapped)
 *   lut_meta[chunk_idx] -> { rc }                (cold path only)
 *
 * Both are MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE mmaps. The kernel
 * demand-pages individual host pages on first write, so the physical footprint
 * scales with live chunks rather than virtual capacity. With a 48-bit VA and
 * 2 MiB granularity that is 1 GiB (lut_phys) + 512 MiB (lut_meta) of virtual
 * reservation, but typically a few hundred KiB resident.
 *
 * The split keeps the hot path flat-array-of-u64: `_virt_to_phys` reads
 * `lut_phys[chunk_idx]` only and never touches `lut_meta`.
 *
 * Caveat: Hardware requirements
 * -----------------------------
 *
 * Same as cudamem_heap: requires a GPU with PCIe P2P DMA support and a BAR1
 * region at least as large as device memory. cudamem_config_init() verifies the
 * BAR1 precondition. Additionally, the chunk-cache assumes the BAR1 page-table
 * maps each alloc_granularity-sized export as one contiguous IOVA range; on
 * hardware violating this, _add returns -EOPNOTSUPP after detecting a
 * non-contiguous LUT.
 *
 * Caveat: Single-GPU registry
 * ---------------------------
 *
 * One registry describes one device. Multi-GPU usage requires one
 * cudamem_mapping_registry per GPU (and one cudamem_config, one cudamem_heap
 * each).
 *
 * Caveat: Virtual address width
 * -----------------------------
 *
 * The LUT capacity assumes user vaddrs fit in CUDAMEM_MAPPING_VA_BITS bits.
 * 48 bits covers x86-64 and ARM64 default page-table widths. Systems with
 * LA57 or 52-bit ARM64 user VA may return vaddrs above this bound; on those
 * systems the constant must be raised at compile time.
 *
 * Caveat: Virtual memory reservation
 * ----------------------------------
 *
 * The two LUTs reserve 1 GiB (lut_phys) + 512 MiB (lut_meta) of virtual
 * address space at 48-bit VA / 2 MiB granularity. They are MAP_NORESERVE,
 * so no swap is committed and physical RSS scales with live chunks. On
 * systems configured for strict accounting (vm.overcommit_memory=2) or
 * inside cgroups whose memory.max counts virtual reservation, registry_init
 * may return -ENOMEM. Workarounds are to relax the accounting policy or to
 * lower CUDAMEM_MAPPING_VA_BITS to bound the reservation.
 *
 * Sentinel: lut_phys[idx] == 0
 * ----------------------------
 *
 * `_virt_to_phys` treats `lut_phys[idx] == 0` as "chunk not registered".
 * Phys 0 is a theoretically valid BAR1 base, but PCIe BAR1 is allocated by
 * the host bridge into nonzero IOVA space in practice. Code paths that
 * mutate state (`_add`, `_remove`, `_clear`) consult `lut_meta[idx].rc`
 * directly and do not depend on this sentinel.
 */

/**
 * Width of the user virtual address space, in bits, used to size the chunk
 * LUTs.
 */
#define CUDAMEM_MAPPING_VA_BITS 48

/**
 * Per-chunk LUT entry (cold-path only).
 *
 * Tracks how many registrations cover this chunk so the lut_phys slot can be
 * cleared when the last one is removed. The dma-buf that owns the chunk's
 * BAR1 export lives on a `cudamem_mapping_handle`, not here.
 */
struct cudamem_mapping_chunk_meta {
	uint32_t rc; ///< Refcount of registrations whose floored range covers this chunk
};

/**
 * Refcounted dma-buf shared by all registrations whose floored range is
 * contained in the handle's range.
 *
 * One handle owns one fd plus its dmabuf attachment. A new registration
 * whose floored range is fully contained in `[floored_va, floored_va +
 * floored_size)` (the same range, or any sub-range) bumps `rc` instead of
 * exporting a new dma-buf. When `rc` reaches zero the dma-buf is detached
 * (closing the fd) and the handle is unlinked from the registry.
 */
struct cudamem_mapping_handle {
	uint64_t floored_va;                  ///< alloc_granularity-floored vaddr of the export
	size_t floored_size;                  ///< chunk-multiple size of the export, in bytes
	uint32_t rc;                          ///< Number of registrations sharing this dma-buf
	struct dmabuf attach;                 ///< dma-buf attachment (one fd)
	struct cudamem_mapping_handle *next;  ///< Linkage in registry->handles
};

/**
 * Per-registration metadata.
 *
 * Tracks one (vaddr, size) registration and the handle that exports its
 * floored range, so `_remove(vaddr)` can locate the chunks to deref and the
 * handle to drop.
 */
struct cudamem_mapping_registration {
	uint64_t vaddr;                            ///< Virtual address of the registered range
	size_t size;                               ///< Size of the registered range in bytes
	struct cudamem_mapping_handle *handle;     ///< Shared handle owning the dma-buf
	struct cudamem_mapping_registration *next; ///< List linkage owned by the registry
};

/**
 * Registry of all registrations for one CUDA device.
 *
 * Owns two demand-paged LUTs covering the full user VA space at chunk
 * granularity: lut_phys (1 GiB virtual at 48-bit VA / 2 MiB gran) for the hot
 * path and lut_meta (512 MiB virtual at 48-bit VA / 2 MiB gran) for the cold
 * path, plus the registration list and the handle list.
 *
 * `gran_shift` and `gran_mask` are derived from `cudamem_config.alloc_granularity`
 * at init time and cached here so the LUT-only paths (`_virt_to_phys`,
 * `_remove`, `_clear`) need no config reference. `_add` still takes the config
 * as a separate argument because handle population calls into CUDA.
 */
struct cudamem_mapping_registry {
	int gran_shift;                              ///< alloc_granularity expressed as a power of two
	uint64_t gran_mask;                          ///< alloc_granularity - 1, for chunk-offset masking
	size_t lut_capacity;                         ///< Number of slots in each LUT
	uint64_t *lut_phys;                          ///< chunk_idx -> phys_base; mmap-backed
	struct cudamem_mapping_chunk_meta *lut_meta; ///< chunk_idx -> meta; mmap-backed
	struct cudamem_mapping_registration *list;   ///< Owned list of registration metadata
	struct cudamem_mapping_handle *handles;      ///< Owned list of dma-buf handles
};

/**
 * Initialize a mapping registry for the given config.
 *
 * Reserves two demand-paged virtual ranges (lut_phys and lut_meta), each
 * sized to (1 << CUDAMEM_MAPPING_VA_BITS) / alloc_granularity slots. No
 * physical memory is committed until chunks are registered.
 *
 * Caches `gran_shift`, `gran_mask`, and `lut_capacity` derived from the
 * config; the registry retains no pointer to the config struct, so the
 * caller is free to discard it after init. `cudamem_mapping_add` takes the
 * config as a separate argument since populating a handle needs the full
 * config (page size, etc.).
 *
 * @return 0 on success, negative errno on failure.
 */
static inline int
cudamem_mapping_registry_init(struct cudamem_mapping_registry *registry,
			      struct cudamem_config *config)
{
	size_t phys_bytes, meta_bytes;

	if (!registry || !config) {
		return -EINVAL;
	}

	registry->gran_shift = config->alloc_granularity_shift;
	registry->gran_mask = (uint64_t)config->alloc_granularity - 1;
	registry->lut_capacity = (1ULL << CUDAMEM_MAPPING_VA_BITS) >> registry->gran_shift;
	registry->list = NULL;
	registry->handles = NULL;

	phys_bytes = registry->lut_capacity * sizeof(*registry->lut_phys);
	registry->lut_phys = mmap(NULL, phys_bytes, PROT_READ | PROT_WRITE,
				  MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
	if (registry->lut_phys == MAP_FAILED) {
		UPCIE_DEBUG("FAILED: mmap(lut_phys, %zu); errno: %d", phys_bytes, errno);
		registry->lut_phys = NULL;
		return -ENOMEM;
	}

	meta_bytes = registry->lut_capacity * sizeof(*registry->lut_meta);
	registry->lut_meta = mmap(NULL, meta_bytes, PROT_READ | PROT_WRITE,
				  MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
	if (registry->lut_meta == MAP_FAILED) {
		UPCIE_DEBUG("FAILED: mmap(lut_meta, %zu); errno: %d", meta_bytes, errno);
		munmap(registry->lut_phys, phys_bytes);
		registry->lut_phys = NULL;
		registry->lut_meta = NULL;
		return -ENOMEM;
	}

	return 0;
}

/**
 * Decrement rc on chunks [chunk_first, chunk_first + chunk_cnt) and clear the
 * lut_phys slot for each chunk whose rc transitions to zero.
 *
 * Chunks with rc == 0 on entry are skipped, making the helper safe for
 * partial unwind paths (where some chunks in the range were never bumped).
 */
static inline void
cudamem_mapping_chunk_deref(struct cudamem_mapping_registry *registry, size_t chunk_first,
			    size_t chunk_cnt)
{
	for (size_t k = 0; k < chunk_cnt; ++k) {
		const size_t idx = chunk_first + k;
		struct cudamem_mapping_chunk_meta *cm = &registry->lut_meta[idx];

		if (cm->rc == 0) {
			continue;
		}
		cm->rc--;
		if (cm->rc == 0) {
			registry->lut_phys[idx] = 0;
		}
	}
}

/**
 * Detach all dma-bufs and free registration metadata.
 *
 * Walks the handle list and detaches each dma-buf, walks the registration
 * list and frees each node, and zeroes out the LUT slots covered by any
 * handle. The LUT mmaps stay reserved.
 */
static inline void
cudamem_mapping_clear(struct cudamem_mapping_registry *registry)
{
	struct cudamem_mapping_handle *hnext;
	struct cudamem_mapping_registration *next;

	if (!registry) {
		return;
	}

	const int gran_shift = registry->gran_shift;

	for (struct cudamem_mapping_handle *h = registry->handles; h; h = hnext) {
		const size_t chunk_first = (size_t)(h->floored_va >> gran_shift);
		const size_t chunk_cnt = h->floored_size >> gran_shift;

		for (size_t k = 0; k < chunk_cnt; ++k) {
			registry->lut_phys[chunk_first + k] = 0;
			registry->lut_meta[chunk_first + k].rc = 0;
		}

		hnext = h->next;
		dmabuf_detach(&h->attach);
		free(h);
	}
	registry->handles = NULL;

	for (struct cudamem_mapping_registration *m = registry->list; m; m = next) {
		next = m->next;
		free(m);
	}
	registry->list = NULL;
}

/**
 * Tear down a registry, releasing the LUT mmaps and detaching all dma-bufs.
 */
static inline void
cudamem_mapping_registry_term(struct cudamem_mapping_registry *registry)
{
	if (!registry) {
		return;
	}

	cudamem_mapping_clear(registry);

	if (registry->lut_phys) {
		munmap(registry->lut_phys, registry->lut_capacity * sizeof(*registry->lut_phys));
		registry->lut_phys = NULL;
	}
	if (registry->lut_meta) {
		munmap(registry->lut_meta, registry->lut_capacity * sizeof(*registry->lut_meta));
		registry->lut_meta = NULL;
	}
	registry->lut_capacity = 0;
}

/**
 * Look up an existing handle that fully contains the given floored range.
 *
 * Walks the handles list. For each handle:
 *   - disjoint from the new range: skip.
 *   - new range is fully contained in handle's range (or matches exactly):
 *     return it via *out and 0.
 *   - any other overlap (handle is contained in new range, or partial
 *     overlap): return -EEXIST. New registrations whose floored range
 *     partially overlaps an existing handle would require either splitting
 *     the existing handle or holding two dma-bufs over the same chunks; both
 *     are out of scope for this registry.
 *
 * If no handle overlaps the new range at all, *out is set to NULL and 0 is
 * returned; the caller should populate a fresh handle.
 *
 * @return 0 on success (with *out set, possibly to NULL), -EEXIST on
 *         non-containing overlap.
 */
static inline int
cudamem_mapping_handle_lookup(struct cudamem_mapping_registry *registry, uint64_t floored_va,
			      size_t floored_size, struct cudamem_mapping_handle **out)
{
	const uint64_t n_end = floored_va + floored_size;

	*out = NULL;
	for (struct cudamem_mapping_handle *h = registry->handles; h; h = h->next) {
		const uint64_t h_end = h->floored_va + h->floored_size;

		if (n_end <= h->floored_va || floored_va >= h_end) {
			continue;
		}
		if (floored_va >= h->floored_va && n_end <= h_end) {
			*out = h;
			return 0;
		}
		UPCIE_DEBUG("FAILED: floored range [0x%" PRIx64 ", 0x%" PRIx64
			    ") overlaps existing handle [0x%" PRIx64 ", 0x%" PRIx64
			    ") without containment",
			    floored_va, n_end, h->floored_va, h_end);
		return -EEXIST;
	}
	return 0;
}

/**
 * Populate a freshly-allocated handle from CUDA.
 *
 * Calls cuMemGetHandleForAddressRange for the handle's full floored range,
 * attaches the dma-buf, fetches the host-page LUT, verifies that each chunk
 * within the range is internally contiguous (BAR1 large-page assumption), and
 * writes one phys_base per chunk into `registry->lut_phys`. Cross-chunk
 * boundaries are not required to be contiguous.
 *
 * On failure no registry state is mutated and the handle's `attach` is left
 * unset. On success the handle's `attach` holds one fd.
 *
 * @return 0 on success, negative errno on failure.
 */
static inline int
cudamem_mapping_handle_populate(struct cudamem_mapping_registry *registry,
				struct cudamem_mapping_handle *handle,
				struct cudamem_config *config)
{
	const size_t pagesize = (size_t)config->pagesize;
	const int pagesize_shift = config->pagesize_shift;
	const size_t gran = config->alloc_granularity;
	const int gran_shift = registry->gran_shift;
	const size_t chunk_first = (size_t)(handle->floored_va >> gran_shift);
	const size_t chunk_cnt = handle->floored_size >> gran_shift;
	const size_t pages_per_chunk = gran >> pagesize_shift;
	const size_t nphys = handle->floored_size >> pagesize_shift;
	uint64_t *tmp = NULL;
	int dmabuf_fd = -1;
	int err;
	CUresult cr;

	tmp = calloc(nphys, sizeof(*tmp));
	if (!tmp) {
		return -ENOMEM;
	}

	cr = cuMemGetHandleForAddressRange(&dmabuf_fd, (CUdeviceptr)handle->floored_va,
					   handle->floored_size,
					   CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
	if (cr != CUDA_SUCCESS) {
		UPCIE_DEBUG("FAILED: cuMemGetHandleForAddressRange(0x%" PRIx64 ", %zu), cr: %d",
			    handle->floored_va, handle->floored_size, cr);
		err = -EIO;
		goto err_free;
	}

	err = dmabuf_attach(dmabuf_fd, &handle->attach);
	if (err) {
		UPCIE_DEBUG("FAILED: dmabuf_attach(), err: %d", err);
		close(dmabuf_fd);
		goto err_free;
	}

	err = dmabuf_get_lut(&handle->attach, nphys, tmp, (int)pagesize);
	if (err) {
		UPCIE_DEBUG("FAILED: dmabuf_get_lut(), err: %d", err);
		goto err_detach;
	}

	for (size_t k = 0; k < chunk_cnt; ++k) {
		const size_t base_idx = k * pages_per_chunk;

		for (size_t i = 1; i < pages_per_chunk; ++i) {
			if (tmp[base_idx + i] != tmp[base_idx] + i * pagesize) {
				UPCIE_DEBUG("FAILED: chunk LUT not contiguous at chunk=%zu page=%zu;"
					    " phys[base]=0x%" PRIx64 " phys[base+%zu]=0x%" PRIx64,
					    k, i, tmp[base_idx], i, tmp[base_idx + i]);
				err = -EOPNOTSUPP;
				goto err_detach;
			}
		}
	}

	for (size_t k = 0; k < chunk_cnt; ++k) {
		registry->lut_phys[chunk_first + k] = tmp[k * pages_per_chunk];
	}

	free(tmp);
	return 0;

err_detach:
	dmabuf_detach(&handle->attach);
err_free:
	free(tmp);
	return err;
}

/**
 * Register an externally-allocated CUDA range with the given registry.
 *
 * Walks the alloc_granularity-aligned chunks intersecting [vaddr, vaddr +
 * nbytes), bumps per-chunk rc, and either reuses an existing dma-buf handle
 * (when the floored range is contained in one already in the registry) or
 * exports a new dma-buf via cuMemGetHandleForAddressRange for the full
 * floored range. The new-handle export is one CUDA call regardless of how
 * many chunks the range spans. Floored ranges that partially overlap an
 * existing handle without containment are rejected with -EEXIST.
 *
 * `vaddr` and `nbytes` may have arbitrary byte alignment; the chunk cache
 * resolves at byte granularity. Note that downstream consumers may impose
 * stricter alignment, e.g., NVMe PRP construction requires host-page-aligned
 * buffer addresses (asserted in nvme_request_prep_command_prps_*_cuda_mapped).
 *
 * `config` must describe the same device the registry was initialized with;
 * it is consulted only to populate new handles (host page size for the
 * dma-buf LUT and granularity for the CUDA handle range). It is not retained.
 *
 * If non-NULL, `*out` is set to the newly-allocated registration node on
 * success.
 *
 * NOTE: Set up CUDA Driver (cuInit()) and CUDA Context (cuCtxCreate())
 * before calling this function.
 *
 * @return 0 on success, negative errno on failure. -EINVAL if the vaddr
 *         range exceeds the LUT's chunk_idx capacity (raise
 *         CUDAMEM_MAPPING_VA_BITS). -EEXIST if the floored range partially
 *         overlaps an existing handle without containment. -EOPNOTSUPP if
 *         the BAR1 contiguity assumption is violated for some chunk.
 */
static inline int
cudamem_mapping_add(struct cudamem_mapping_registry *registry, struct cudamem_config *config,
		    void *vaddr, size_t nbytes, struct cudamem_mapping_registration **out)
{
	struct cudamem_mapping_registration *m = NULL;
	struct cudamem_mapping_handle *handle = NULL;
	int new_handle = 0;
	int err;

	if (!registry || !config || !vaddr || !nbytes) {
		return -EINVAL;
	}

	const uint64_t mask = registry->gran_mask;
	const int gran_shift = registry->gran_shift;
	const uint64_t va = (uint64_t)vaddr;
	const uint64_t floored_va = va & ~mask;
	const size_t chunk_first = (size_t)(floored_va >> gran_shift);
	const size_t chunk_cnt = (size_t)(((va & mask) + nbytes + mask) >> gran_shift);
	const size_t floored_size = chunk_cnt << gran_shift;

	if (chunk_first + chunk_cnt > registry->lut_capacity) {
		UPCIE_DEBUG(
			"FAILED: vaddr range exceeds LUT capacity; raise CUDAMEM_MAPPING_VA_BITS");
		return -EINVAL;
	}

	m = calloc(1, sizeof(*m));
	if (!m) {
		return -ENOMEM;
	}
	m->vaddr = va;
	m->size = nbytes;

	err = cudamem_mapping_handle_lookup(registry, floored_va, floored_size, &handle);
	if (err) {
		goto err_free_m;
	}
	if (!handle) {
		handle = calloc(1, sizeof(*handle));
		if (!handle) {
			err = -ENOMEM;
			goto err_free_m;
		}
		handle->floored_va = floored_va;
		handle->floored_size = floored_size;
		handle->rc = 0;
		new_handle = 1;

		err = cudamem_mapping_handle_populate(registry, handle, config);
		if (err) {
			free(handle);
			goto err_free_m;
		}
	}

	for (size_t k = 0; k < chunk_cnt; ++k) {
		registry->lut_meta[chunk_first + k].rc++;
	}

	handle->rc++;
	if (new_handle) {
		handle->next = registry->handles;
		registry->handles = handle;
	}
	m->handle = handle;

	m->next = registry->list;
	registry->list = m;
	if (out) {
		*out = m;
	}

	return 0;

err_free_m:
	free(m);
	return err;
}

/**
 * Drop a handle reference and detach its dma-buf when the last reference goes
 * away.
 */
static inline void
cudamem_mapping_handle_deref(struct cudamem_mapping_registry *registry,
			     struct cudamem_mapping_handle *handle)
{
	if (handle->rc == 0) {
		return;
	}
	handle->rc--;
	if (handle->rc != 0) {
		return;
	}

	for (struct cudamem_mapping_handle **hp = &registry->handles; *hp; hp = &(*hp)->next) {
		if (*hp == handle) {
			*hp = handle->next;
			break;
		}
	}
	dmabuf_detach(&handle->attach);
	free(handle);
}

/**
 * Remove a registration from the registry.
 *
 * Finds the registration with the given vaddr in the list, decrements rc on
 * each chunk it covered (clearing lut_phys slots whose rc reaches zero), and
 * drops a reference to its handle (detaching the dma-buf when the last
 * sharing registration goes away).
 *
 * @return 0 on success, -EINVAL if no registration with that vaddr exists.
 */
static inline int
cudamem_mapping_remove(struct cudamem_mapping_registry *registry, void *vaddr)
{
	if (!registry) {
		return -EINVAL;
	}

	const uint64_t key = (uint64_t)vaddr;
	const int gran_shift = registry->gran_shift;
	const uint64_t mask = registry->gran_mask;

	for (struct cudamem_mapping_registration **prev = &registry->list, *m = registry->list; m;
	     prev = &m->next, m = m->next) {
		if (m->vaddr == key) {
			const size_t chunk_first = (size_t)(m->vaddr >> gran_shift);
			const size_t chunk_cnt =
				(size_t)(((m->vaddr & mask) + m->size + mask) >> gran_shift);

			cudamem_mapping_chunk_deref(registry, chunk_first, chunk_cnt);
			cudamem_mapping_handle_deref(registry, m->handle);

			*prev = m->next;
			free(m);
			return 0;
		}
	}

	return -EINVAL;
}

/**
 * Resolve a CUDA virtual address registered with the registry.
 *
 * O(1): one LUT load.
 *
 * @return 0 on success, -EINVAL if `virt` is not in a registered chunk.
 */
static inline int
cudamem_mapping_virt_to_phys(struct cudamem_mapping_registry *registry, void *virt, uint64_t *phys)
{
	if (!registry || !virt || !phys) {
		return -EINVAL;
	}

	const uint64_t va = (uint64_t)virt;
	const int gran_shift = registry->gran_shift;
	const uint64_t mask = registry->gran_mask;
	const size_t idx = (size_t)(va >> gran_shift);
	uint64_t base;

	if (idx >= registry->lut_capacity) {
		return -EINVAL;
	}

	base = registry->lut_phys[idx];
	if (base == 0) {
		return -EINVAL;
	}

	*phys = base + (va & mask);
	return 0;
}
