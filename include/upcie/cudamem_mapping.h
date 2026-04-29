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
 * Lookup design
 * -------------
 *
 * Registered buffers share a single contiguous LUT sized to total device
 * memory at host-page granularity: `cap = device_memsize / pagesize` entries
 * of 64 bits each. The LUT is anchored at `va_base`, the lowest registered
 * virtual address seen so far. cudamem_mapping_virt_to_phys() resolves any
 * vaddr in O(1) via
 *
 *     phys = phys_lut[(vaddr - va_base) >> pagesize_shift] + (vaddr & (pagesize - 1))
 *
 * with `phys_lut[i] == 0` indicating "no mapping covers this page".
 *
 * Host-page granularity (typically 4 KiB) is the natural floor: dma-buf
 * iterates at host-page granularity at minimum, NVMe PRP entries are
 * host-page aligned, and PCIe P2P targets have no sub-host-page semantics.
 * Going coarser (e.g., 64 KiB device pages) would shrink the LUT but force
 * callers to round up further, which only the heap (which controls
 * allocation) can do for free.
 *
 * `va_base` is a lower bound on the live vaddrs: on cudamem_mapping_add(),
 * if the new vaddr is below the current va_base, va_base is lowered to it
 * and the LUT is shifted up via memmove. va_base is never raised; _remove
 * leaves it as-is (it remains a valid lower bound). The shift is O(cap)
 * but on a cold path; the hot path (virt_to_phys) is one load.
 *
 * Per-mapping metadata (vaddr, size, dma-buf attachment) lives in a singly
 * linked list owned by the registry; it is only walked by _remove and
 * _clear, never by _add or _virt_to_phys.
 *
 * Caveat: Hardware requirements
 * -----------------------------
 *
 * Same as cudamem_heap: requires a GPU with PCIe P2P DMA support and a BAR1
 * region at least as large as device memory. cudamem_config_init() verifies
 * the BAR1 precondition.
 *
 * Caveat: Single-GPU registry
 * ---------------------------
 *
 * One registry describes one device. Multi-GPU usage requires one
 * cudamem_mapping_registry per GPU (and one cudamem_config, one cudamem_heap
 * each).
 */

/**
 * Per-mapping metadata for a registered CUDA buffer
 *
 * The phys entries themselves live in the registry's shared LUT, not here.
 * This struct only holds what's needed to undo the registration: the vaddr
 * (match key), the size (LUT slot range), the dma-buf attachment, and a list
 * link.
 */
struct cudamem_mapping {
	uint64_t vaddr;               ///< Virtual address of the registered range
	size_t size;                  ///< Size of the registered range in bytes
	struct dmabuf dmabuf;         ///< dma-buf attachment for the range
	struct cudamem_mapping *next; ///< List linkage owned by the registry
};

/**
 * Registry of all registered cudamem_mappings for one CUDA device
 *
 * Owns the shared phys_lut (sized to device memory at init) and the linked
 * list of mapping metadata.
 */
struct cudamem_mapping_registry {
	struct cudamem_config *config; ///< Device memory configuration
	uint64_t va_base;              ///< Lower bound on live vaddrs; UINT64_MAX when empty
	size_t cap;                    ///< Number of entries in phys_lut
	uint64_t *phys_lut; ///< phys_lut[i] = phys of (va_base + i*pagesize); 0 == unmapped
	struct cudamem_mapping *list; ///< Owned list of mapping metadata
};

/**
 * Initialize a mapping registry for the given config.
 *
 * Allocates a zero-initialized phys_lut sized to total device memory. The
 * registry starts empty (va_base = UINT64_MAX).
 *
 * NOTE: `config` must remain valid for the lifetime of the registry.
 *
 * @return 0 on success, negative errno on failure.
 */
static inline int
cudamem_mapping_registry_init(struct cudamem_mapping_registry *registry,
			      struct cudamem_config *config)
{
	if (!registry || !config) {
		return -EINVAL;
	}

	registry->config = config;
	registry->cap = config->device_memsize / (size_t)config->pagesize;
	registry->va_base = UINT64_MAX;
	registry->list = NULL;

	registry->phys_lut = calloc(registry->cap, sizeof(uint64_t));
	if (!registry->phys_lut) {
		UPCIE_DEBUG("FAILED: calloc(phys_lut, cap=%zu)", registry->cap);
		return -ENOMEM;
	}

	return 0;
}

/**
 * Detach all registered mappings, free metadata, and reset the registry.
 *
 * The phys_lut is zeroed and va_base reset to UINT64_MAX.
 */
static inline void
cudamem_mapping_clear(struct cudamem_mapping_registry *registry)
{
	struct cudamem_mapping *next;

	if (!registry) {
		return;
	}

	for (struct cudamem_mapping *m = registry->list; m; m = next) {
		next = m->next;
		dmabuf_detach(&m->dmabuf);
		free(m);
	}
	registry->list = NULL;

	if (registry->phys_lut) {
		memset(registry->phys_lut, 0, registry->cap * sizeof(uint64_t));
	}
	registry->va_base = UINT64_MAX;
}

/**
 * Tear down a registry, freeing the LUT and detaching all mappings.
 */
static inline void
cudamem_mapping_registry_term(struct cudamem_mapping_registry *registry)
{
	if (!registry) {
		return;
	}

	cudamem_mapping_clear(registry);
	free(registry->phys_lut);
	registry->phys_lut = NULL;
	registry->cap = 0;
}

/**
 * Allocate a CUDA buffer aligned to host pagesize.
 *
 * cuMemAlloc does not guarantee host-page alignment of the returned vaddr
 * (small allocations may be slab-suballocated at sub-page stride). This
 * helper over-allocates by one host page and returns:
 *
 *  - `raw`           - the cuMemAlloc handle, used for cuMemFree.
 *  - `aligned`       - the pagesize-aligned start used for registration.
 *  - `aligned_nbytes`- nbytes rounded up to pagesize, the size to pass
 *                      to cudamem_mapping_add().
 *
 * The caller frees with cuMemFree(*raw); cudamem_mapping_remove() then
 * detaches the dma-buf for the aligned subrange.
 *
 * @return 0 on success, negative errno on failure.
 */
static inline int
cudamem_alloc_aligned(struct cudamem_config *config, size_t nbytes, CUdeviceptr *raw,
		      void **aligned, size_t *aligned_nbytes)
{
	const size_t page = (size_t)config->pagesize;
	size_t rounded = (nbytes + page - 1) & ~(page - 1);
	CUdeviceptr p;
	CUresult cr;

	if (!config || !nbytes || !raw || !aligned) {
		return -EINVAL;
	}

	cr = cuMemAlloc(&p, rounded + page);
	if (cr != CUDA_SUCCESS) {
		UPCIE_DEBUG("FAILED: cuMemAlloc(%zu), cr: %d", rounded + page, cr);
		return -ENOMEM;
	}

	*raw = p;
	*aligned = (void *)(((uintptr_t)p + page - 1) & ~(page - 1));
	if (aligned_nbytes) {
		*aligned_nbytes = rounded;
	}

	return 0;
}

/**
 * Register an externally-allocated CUDA range with the given registry.
 *
 * Obtains a dma-buf for [vaddr, vaddr + nbytes) via
 * cuMemGetHandleForAddressRange(), attaches it, and writes its physical
 * addresses into the registry's shared phys_lut.
 *
 * `vaddr` and `nbytes` must be aligned to `config->pagesize` (host page,
 * typically 4 KiB). Callers who want to start from a `cuMemAlloc` return can
 * use cudamem_alloc_aligned() — it over-allocates by one host page and
 * returns the aligned subrange.
 *
 * If the new vaddr is below the current va_base, va_base is lowered to it
 * and the LUT is shifted up via memmove (O(cap), but only on cold path).
 *
 * NOTE: Set up CUDA Driver (cuInit()) and CUDA Context (cuCtxCreate())
 * before calling this function.
 *
 * @return 0 on success, negative errno on failure. -EOVERFLOW if the new
 *         mapping would push the live VA span beyond device memory size.
 */
static inline int
cudamem_mapping_add(struct cudamem_mapping_registry *registry, void *vaddr, size_t nbytes,
		    struct cudamem_mapping **out)
{
	const uint64_t va = (uint64_t)vaddr;
	const size_t page = (size_t)registry->config->pagesize;
	const size_t cap_bytes = registry->cap * page;
	struct cudamem_mapping *m = NULL;
	uint64_t new_va_base;
	size_t idx, npages;
	int dmabuf_fd = -1, err;
	CUresult cr;

	if (!registry || !vaddr || !nbytes) {
		return -EINVAL;
	}
	if (va % page || nbytes % page) {
		UPCIE_DEBUG("FAILED: vaddr/nbytes not aligned to pagesize(%zu)", page);
		return -EINVAL;
	}

	m = calloc(1, sizeof(*m));
	if (!m) {
		return -ENOMEM;
	}
	m->vaddr = va;
	m->size = nbytes;

	cr = cuMemGetHandleForAddressRange(&dmabuf_fd, (CUdeviceptr)vaddr, nbytes,
					   CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
	if (cr != CUDA_SUCCESS) {
		UPCIE_DEBUG("FAILED: cuMemGetHandleForAddressRange(), cr: %d", cr);
		err = -EIO;
		goto err_free;
	}

	err = dmabuf_attach(dmabuf_fd, &m->dmabuf);
	if (err) {
		UPCIE_DEBUG("FAILED: dmabuf_attach(), err: %d", err);
		close(dmabuf_fd);
		goto err_free;
	}

	/* va_base is a lower bound on the live vaddrs: lower it when we see a
	 * smaller vaddr, never raise it. _remove leaves it as-is. The
	 * empty-registry case falls into the first branch via va_base ==
	 * UINT64_MAX. */
	new_va_base = (va < registry->va_base) ? va : registry->va_base;

	if ((va - new_va_base) + nbytes > cap_bytes) {
		UPCIE_DEBUG("FAILED: live VA span exceeds device memory; cap_bytes(%zu)",
			    cap_bytes);
		err = -EOVERFLOW;
		goto err_detach;
	}

	if (new_va_base < registry->va_base && registry->va_base != UINT64_MAX) {
		uint64_t *lut = registry->phys_lut;
		size_t delta = (size_t)((registry->va_base - new_va_base) / page);
		if (delta > registry->cap) {
			delta = registry->cap;
		}
		memmove(lut + delta, lut, (registry->cap - delta) * sizeof(uint64_t));
		memset(lut, 0, delta * sizeof(uint64_t));
	}
	registry->va_base = new_va_base;

	idx = (size_t)((va - new_va_base) / page);
	npages = nbytes / page;
	err = dmabuf_get_lut(&m->dmabuf, npages, registry->phys_lut + idx, (int)page);
	if (err) {
		UPCIE_DEBUG("FAILED: dmabuf_get_lut(), err: %d", err);
		memset(registry->phys_lut + idx, 0, npages * sizeof(uint64_t));
		goto err_detach;
	}

	m->next = registry->list;
	registry->list = m;
	if (out) {
		*out = m;
	}

	return 0;

err_detach:
	dmabuf_detach(&m->dmabuf);
err_free:
	free(m);
	return err;
}

/**
 * Remove the mapping with the given vaddr from the registry.
 *
 * Zeros the corresponding phys_lut slots, detaches the dma-buf, and frees
 * the mapping struct. va_base is reset to UINT64_MAX if the registry
 * becomes empty; otherwise it is left as-is, since it remains a valid lower
 * bound on the live vaddrs.
 *
 * @return 0 on success, -EINVAL if no mapping with that vaddr is registered.
 */
static inline int
cudamem_mapping_remove(struct cudamem_mapping_registry *registry, void *vaddr)
{
	const uint64_t key = (uint64_t)vaddr;
	const size_t page = (size_t)registry->config->pagesize;

	if (!registry) {
		return -EINVAL;
	}

	for (struct cudamem_mapping **prev = &registry->list, *m = registry->list; m;
	     prev = &m->next, m = m->next) {
		if (m->vaddr == key) {
			size_t idx = (size_t)((m->vaddr - registry->va_base) / page);
			size_t npages = m->size / page;

			memset(registry->phys_lut + idx, 0, npages * sizeof(uint64_t));

			*prev = m->next;
			dmabuf_detach(&m->dmabuf);
			free(m);

			if (!registry->list) {
				registry->va_base = UINT64_MAX;
			}
			return 0;
		}
	}

	return -EINVAL;
}

/**
 * Resolve a CUDA virtual address registered with the registry.
 *
 * O(1): one bounds check and one LUT load.
 *
 * @return 0 on success, -EINVAL if `virt` is not in the LUT window or the
 *         page is unmapped.
 */
static inline int
cudamem_mapping_virt_to_phys(struct cudamem_mapping_registry *registry, void *virt, uint64_t *phys)
{
	const uint64_t va = (uint64_t)virt;
	const int shift = registry->config->pagesize_shift;
	const uint64_t mask = ((uint64_t)1 << shift) - 1;
	uint64_t addr, offset;
	size_t idx;

	if (!registry || !virt || !phys) {
		return -EINVAL;
	}
	if (va < registry->va_base) {
		return -EINVAL;
	}

	offset = va - registry->va_base;
	idx = (size_t)(offset >> shift);
	if (idx >= registry->cap) {
		return -EINVAL;
	}

	addr = registry->phys_lut[idx];
	if (addr == 0) {
		return -EINVAL;
	}

	*phys = addr + (offset & mask);
	return 0;
}
