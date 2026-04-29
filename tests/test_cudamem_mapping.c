// SPDX-License-Identifier: BSD-3-Clause

#include <upcie/upcie_cuda.h>
#include <cuda.h>

#define EXPECT_EQ(label, got, want)                                                      \
	do {                                                                             \
		long long _g = (long long)(got);                                         \
		long long _w = (long long)(want);                                        \
		if (_g != _w) {                                                          \
			printf("# FAILED: %s; got(%lld) want(%lld)\n", (label), _g, _w); \
			return 1;                                                        \
		}                                                                        \
	} while (0)

static int
test_add_remove_clear(struct cudamem_mapping_registry *registry, struct cudamem_config *config)
{
	const size_t page = (size_t)config->device_pagesize;
	CUdeviceptr raw = 0;
	void *aligned = NULL;
	size_t aligned_nbytes = 0;
	uint64_t phys = 0;
	int err;

	printf("# test_add_remove_clear\n");

	err = cudamem_alloc_aligned(config, 4 * page, &raw, &aligned, &aligned_nbytes);
	EXPECT_EQ("cudamem_alloc_aligned", err, 0);

	err = cudamem_mapping_add(registry, aligned, aligned_nbytes, NULL);
	EXPECT_EQ("cudamem_mapping_add", err, 0);

	err = cudamem_mapping_virt_to_phys(registry, aligned, &phys);
	EXPECT_EQ("virt_to_phys(base)", err, 0);
	if (phys == 0) {
		printf("# FAILED: phys is zero\n");
		return 1;
	}

	err = cudamem_mapping_remove(registry, aligned);
	EXPECT_EQ("cudamem_mapping_remove", err, 0);

	err = cudamem_mapping_virt_to_phys(registry, aligned, &phys);
	EXPECT_EQ("virt_to_phys(after remove)", err, -EINVAL);

	err = cudamem_mapping_add(registry, aligned, aligned_nbytes, NULL);
	EXPECT_EQ("cudamem_mapping_add(re-add)", err, 0);

	cudamem_mapping_clear(registry);
	if (registry->list != NULL) {
		printf("# FAILED: registry->list non-NULL after clear\n");
		return 1;
	}
	if (registry->va_base != UINT64_MAX) {
		printf("# FAILED: registry->va_base not reset after clear\n");
		return 1;
	}

	cuMemFree(raw);
	return 0;
}

static int
test_unaligned_lookup(struct cudamem_mapping_registry *registry, struct cudamem_config *config)
{
	const size_t page = (size_t)config->device_pagesize;
	CUdeviceptr raw = 0;
	void *aligned = NULL;
	size_t aligned_nbytes = 0;
	uint64_t base_phys = 0, off_phys = 0;
	const uint64_t in_page_offset = 1234;
	const uint64_t cross_page_offset = page + 4321;
	int err;

	printf("# test_unaligned_lookup\n");

	err = cudamem_alloc_aligned(config, 4 * page, &raw, &aligned, &aligned_nbytes);
	EXPECT_EQ("cudamem_alloc_aligned", err, 0);

	err = cudamem_mapping_add(registry, aligned, aligned_nbytes, NULL);
	EXPECT_EQ("cudamem_mapping_add", err, 0);

	err = cudamem_mapping_virt_to_phys(registry, aligned, &base_phys);
	EXPECT_EQ("virt_to_phys(base)", err, 0);

	/* Unaligned VA inside the first page must resolve to base_phys + offset. */
	err = cudamem_mapping_virt_to_phys(registry, (uint8_t *)aligned + in_page_offset,
					   &off_phys);
	EXPECT_EQ("virt_to_phys(in-page offset)", err, 0);
	if (off_phys != base_phys + in_page_offset) {
		printf("# FAILED: in-page offset mismatch; got(0x%" PRIx64 ") want(0x%" PRIx64
		       ")\n",
		       off_phys, base_phys + in_page_offset);
		return 1;
	}

	/* Unaligned VA in a later page resolves through a different LUT slot;
	 * within-page offset must still be added. */
	uint64_t page2_base = 0;
	err = cudamem_mapping_virt_to_phys(registry, (uint8_t *)aligned + page, &page2_base);
	EXPECT_EQ("virt_to_phys(page2 base)", err, 0);

	err = cudamem_mapping_virt_to_phys(registry, (uint8_t *)aligned + cross_page_offset,
					   &off_phys);
	EXPECT_EQ("virt_to_phys(cross-page offset)", err, 0);
	if (off_phys != page2_base + (cross_page_offset - page)) {
		printf("# FAILED: cross-page offset mismatch; got(0x%" PRIx64 ") want(0x%" PRIx64
		       ")\n",
		       off_phys, page2_base + (cross_page_offset - page));
		return 1;
	}

	cudamem_mapping_clear(registry);
	cuMemFree(raw);
	return 0;
}

static int
test_misalignment(struct cudamem_mapping_registry *registry, struct cudamem_config *config)
{
	const size_t host_page = (size_t)config->pagesize;
	const size_t dev_page = (size_t)config->device_pagesize;
	CUdeviceptr raw = 0;
	void *aligned = NULL;
	size_t aligned_nbytes = 0;
	int err;

	printf("# test_misalignment\n");

	err = cudamem_alloc_aligned(config, 4 * dev_page, &raw, &aligned, &aligned_nbytes);
	EXPECT_EQ("cudamem_alloc_aligned", err, 0);

	/* Sub-host-page vaddr: rejected. */
	err = cudamem_mapping_add(registry, (uint8_t *)aligned + 1, aligned_nbytes, NULL);
	EXPECT_EQ("add(sub-host-page vaddr)", err, -EINVAL);

	/* Sub-host-page size: rejected. */
	err = cudamem_mapping_add(registry, aligned, aligned_nbytes - 1, NULL);
	EXPECT_EQ("add(sub-host-page size)", err, -EINVAL);

	/* Host-page aligned but sub-device-page aligned: accepted under the
	 * 4K-granular LUT. Skip the first device page to make the start vaddr
	 * itself sub-device-page aligned (only host-page aligned), and also
	 * pick a size that is not a device-page multiple. */
	err = cudamem_mapping_add(registry, (uint8_t *)aligned + host_page,
				  aligned_nbytes - 2 * host_page, NULL);
	EXPECT_EQ("add(host-page aligned, sub-device-page)", err, 0);

	cudamem_mapping_clear(registry);
	cuMemFree(raw);
	return 0;
}

static int
test_lookup_unmapped(struct cudamem_mapping_registry *registry, struct cudamem_config *config)
{
	const size_t page = (size_t)config->device_pagesize;
	CUdeviceptr raw = 0;
	void *aligned = NULL;
	size_t aligned_nbytes = 0;
	uint64_t phys = 0;
	int err;

	printf("# test_lookup_unmapped\n");

	err = cudamem_alloc_aligned(config, 2 * page, &raw, &aligned, &aligned_nbytes);
	EXPECT_EQ("cudamem_alloc_aligned", err, 0);

	err = cudamem_mapping_add(registry, aligned, aligned_nbytes, NULL);
	EXPECT_EQ("cudamem_mapping_add", err, 0);

	/* Below va_base. */
	err = cudamem_mapping_virt_to_phys(registry, (void *)((uint64_t)aligned - page), &phys);
	EXPECT_EQ("virt_to_phys(below va_base)", err, -EINVAL);

	/* Inside LUT window but past the registered range and not covered. */
	void *far = (uint8_t *)aligned + ((size_t)1 << 30);
	if ((uint64_t)far - (uint64_t)aligned < registry->cap * (size_t)config->pagesize) {
		err = cudamem_mapping_virt_to_phys(registry, far, &phys);
		EXPECT_EQ("virt_to_phys(unmapped page in window)", err, -EINVAL);
	}

	cudamem_mapping_clear(registry);
	cuMemFree(raw);
	return 0;
}

static int
test_multiple_mappings(struct cudamem_mapping_registry *registry, struct cudamem_config *config)
{
	const size_t page = (size_t)config->device_pagesize;
	CUdeviceptr raw[3] = {0};
	void *aligned[3] = {0};
	size_t aligned_nbytes[3] = {0};
	uint64_t phys = 0;
	int err;

	printf("# test_multiple_mappings\n");

	for (int i = 0; i < 3; i++) {
		err = cudamem_alloc_aligned(config, (size_t)(i + 1) * page, &raw[i], &aligned[i],
					    &aligned_nbytes[i]);
		EXPECT_EQ("cudamem_alloc_aligned", err, 0);

		err = cudamem_mapping_add(registry, aligned[i], aligned_nbytes[i], NULL);
		EXPECT_EQ("cudamem_mapping_add", err, 0);
	}

	for (int i = 0; i < 3; i++) {
		err = cudamem_mapping_virt_to_phys(registry, aligned[i], &phys);
		EXPECT_EQ("virt_to_phys", err, 0);
		if (phys == 0) {
			printf("# FAILED: phys is zero for mapping %d\n", i);
			return 1;
		}
	}

	/* Remove the middle one and verify the other two still resolve. */
	err = cudamem_mapping_remove(registry, aligned[1]);
	EXPECT_EQ("cudamem_mapping_remove(middle)", err, 0);

	err = cudamem_mapping_virt_to_phys(registry, aligned[1], &phys);
	EXPECT_EQ("virt_to_phys(after remove middle)", err, -EINVAL);

	err = cudamem_mapping_virt_to_phys(registry, aligned[0], &phys);
	EXPECT_EQ("virt_to_phys(0 still mapped)", err, 0);

	err = cudamem_mapping_virt_to_phys(registry, aligned[2], &phys);
	EXPECT_EQ("virt_to_phys(2 still mapped)", err, 0);

	cudamem_mapping_clear(registry);
	for (int i = 0; i < 3; i++) {
		cuMemFree(raw[i]);
	}
	return 0;
}

static int
test_va_base_shift(struct cudamem_mapping_registry *registry, struct cudamem_config *config)
{
	/* Force a downward va_base shift by registering a small allocation, then a
	 * large one further up, then a third one whose VA address is below the
	 * current va_base. Concretely we drive the case via three cuMemAllocs in
	 * sequence; we cannot control the driver's placement, but we register them
	 * in different orders to exercise both the upward and downward shift code
	 * paths in cudamem_mapping_add. */
	const size_t page = (size_t)config->device_pagesize;
	CUdeviceptr raw[3] = {0};
	void *aligned[3] = {0};
	size_t aligned_nbytes[3] = {0};
	uint64_t pre_va_base, post_va_base;
	uint64_t phys = 0;
	int err;

	printf("# test_va_base_shift\n");

	for (int i = 0; i < 3; i++) {
		err = cudamem_alloc_aligned(config, 4 * page, &raw[i], &aligned[i],
					    &aligned_nbytes[i]);
		EXPECT_EQ("cudamem_alloc_aligned", err, 0);
	}

	/* Sort indices by vaddr ascending so we can register the highest first
	 * and force at least one downward va_base shift. */
	int order[3] = {0, 1, 2};
	for (int i = 0; i < 3; i++) {
		for (int j = i + 1; j < 3; j++) {
			if ((uint64_t)aligned[order[i]] < (uint64_t)aligned[order[j]]) {
				int tmp = order[i];
				order[i] = order[j];
				order[j] = tmp;
			}
		}
	}

	/* Highest first. va_base is set to this. */
	err = cudamem_mapping_add(registry, aligned[order[0]], aligned_nbytes[order[0]], NULL);
	EXPECT_EQ("add(highest)", err, 0);
	pre_va_base = registry->va_base;
	if (pre_va_base != (uint64_t)aligned[order[0]]) {
		printf("# FAILED: va_base after first add\n");
		return 1;
	}

	/* Middle. Below current va_base, so va_base shifts down. */
	err = cudamem_mapping_add(registry, aligned[order[1]], aligned_nbytes[order[1]], NULL);
	EXPECT_EQ("add(middle)", err, 0);
	post_va_base = registry->va_base;
	if (post_va_base != (uint64_t)aligned[order[1]]) {
		printf("# FAILED: va_base did not shift down to middle vaddr\n");
		return 1;
	}

	/* Lowest. Triggers another downward shift. */
	err = cudamem_mapping_add(registry, aligned[order[2]], aligned_nbytes[order[2]], NULL);
	EXPECT_EQ("add(lowest)", err, 0);
	if (registry->va_base != (uint64_t)aligned[order[2]]) {
		printf("# FAILED: va_base did not shift down to lowest vaddr\n");
		return 1;
	}

	/* All three must still resolve through the shifted LUT. */
	for (int i = 0; i < 3; i++) {
		err = cudamem_mapping_virt_to_phys(registry, aligned[i], &phys);
		EXPECT_EQ("virt_to_phys after shifts", err, 0);
		if (phys == 0) {
			printf("# FAILED: phys is zero after shift for mapping %d\n", i);
			return 1;
		}
	}

	cudamem_mapping_clear(registry);
	for (int i = 0; i < 3; i++) {
		cuMemFree(raw[i]);
	}
	return 0;
}

static int
test_cumemgethandle_unaligned(struct cudamem_config *config)
{
	const size_t page = (size_t)config->device_pagesize;
	const size_t npages = 4;
	CUdeviceptr raw = 0;
	void *aligned = NULL;
	size_t aligned_nbytes = 0;
	int dmabuf_fd = -1;
	struct dmabuf dmabuf = {0};
	uint64_t baseline_phys[4] = {0};
	uint64_t probe_phys[4] = {0};
	CUresult cr;
	int err;

	printf("# test_cumemgethandle_unaligned\n");

	err = cudamem_alloc_aligned(config, npages * page, &raw, &aligned, &aligned_nbytes);
	EXPECT_EQ("cudamem_alloc_aligned", err, 0);

	/* Aligned baseline: must succeed end-to-end so we have phys to compare against. */
	cr = cuMemGetHandleForAddressRange(&dmabuf_fd, (CUdeviceptr)aligned, aligned_nbytes,
					   CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
	if (cr != CUDA_SUCCESS) {
		printf("# FAILED: aligned baseline cuMemGetHandleForAddressRange; cr(%d)\n", cr);
		cuMemFree(raw);
		return 1;
	}
	err = dmabuf_attach(dmabuf_fd, &dmabuf);
	if (err) {
		printf("# FAILED: aligned baseline dmabuf_attach; err(%d)\n", err);
		close(dmabuf_fd);
		cuMemFree(raw);
		return 1;
	}
	err = dmabuf_get_lut(&dmabuf, npages, baseline_phys, (int)page);
	if (err) {
		printf("# FAILED: aligned baseline dmabuf_get_lut; err(%d)\n", err);
		dmabuf_detach(&dmabuf);
		cuMemFree(raw);
		return 1;
	}
	dmabuf_detach(&dmabuf);
	printf("# baseline phys[0..3]=0x%" PRIx64 " 0x%" PRIx64 " 0x%" PRIx64 " 0x%" PRIx64 "\n",
	       baseline_phys[0], baseline_phys[1], baseline_phys[2], baseline_phys[3]);

	const size_t host_page = (size_t)config->pagesize;

	/* Probe 1: unaligned vaddr (+4096). Walk all the way through dmabuf_get_lut
	 * at both 64K stride (driver-default granularity) and host-page stride (the
	 * granularity a relaxed-alignment LUT would use). */
	cr = cuMemGetHandleForAddressRange(&dmabuf_fd, (CUdeviceptr)((uint8_t *)aligned + 4096),
					   aligned_nbytes - page,
					   CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
	printf("# unaligned vaddr (+4096): cuMemGetHandleForAddressRange cr=%d\n", cr);
	if (cr == CUDA_SUCCESS) {
		memset(&dmabuf, 0, sizeof(dmabuf));
		err = dmabuf_attach(dmabuf_fd, &dmabuf);
		printf("# unaligned vaddr: dmabuf_attach err=%d\n", err);
		if (err == 0) {
			memset(probe_phys, 0, sizeof(probe_phys));
			err = dmabuf_get_lut(&dmabuf, npages - 1, probe_phys, (int)page);
			printf("# unaligned vaddr: 64K-stride dmabuf_get_lut err=%d "
			       "phys[0]=0x%" PRIx64 "\n",
			       err, probe_phys[0]);
			if (err == 0) {
				if (probe_phys[0] == baseline_phys[0]) {
					printf("# unaligned vaddr: driver aligned DOWN (phys[0] "
					       "== baseline phys[0])\n");
				} else if (probe_phys[0] == baseline_phys[1]) {
					printf("# unaligned vaddr: driver aligned UP (phys[0] == "
					       "baseline phys[1])\n");
				} else {
					printf("# unaligned vaddr: phys[0] matches neither "
					       "baseline page\n");
				}
			}

			/* 4K-stride probe: would a host-page-granular LUT work? */
			const size_t npages_4k = ((npages - 1) * page) / host_page;
			uint64_t *probe_4k = calloc(npages_4k, sizeof(uint64_t));
			if (probe_4k) {
				err = dmabuf_get_lut(&dmabuf, npages_4k, probe_4k, (int)host_page);
				printf("# unaligned vaddr: 4K-stride dmabuf_get_lut(npages=%zu) "
				       "err=%d\n",
				       npages_4k, err);
				if (err == 0) {
					printf("# unaligned vaddr 4K phys[0,1,14,15,16]="
					       "0x%" PRIx64 " 0x%" PRIx64 " 0x%" PRIx64
					       " 0x%" PRIx64 " 0x%" PRIx64 "\n",
					       probe_4k[0], probe_4k[1], probe_4k[14],
					       probe_4k[15], probe_4k[16]);
					int monotonic = 1;
					for (size_t i = 1; i < npages_4k; i++) {
						if (probe_4k[i] != probe_4k[i - 1] + host_page) {
							printf("# unaligned vaddr: 4K phys "
							       "discontinuity at i=%zu "
							       "(0x%" PRIx64
							       " vs expected 0x%" PRIx64 ")\n",
							       i, probe_4k[i],
							       probe_4k[i - 1] + host_page);
							monotonic = 0;
							break;
						}
					}
					if (monotonic) {
						printf("# unaligned vaddr: 4K-stride phys is "
						       "stride-faithful across all %zu entries\n",
						       npages_4k);
					}
				}
				free(probe_4k);
			}
			dmabuf_detach(&dmabuf);
		} else {
			close(dmabuf_fd);
		}
	}

	/* Probe 2: unaligned size (-4096). */
	cr = cuMemGetHandleForAddressRange(&dmabuf_fd, (CUdeviceptr)aligned, aligned_nbytes - 4096,
					   CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
	printf("# unaligned size (-4096): cuMemGetHandleForAddressRange cr=%d\n", cr);
	if (cr == CUDA_SUCCESS) {
		memset(&dmabuf, 0, sizeof(dmabuf));
		err = dmabuf_attach(dmabuf_fd, &dmabuf);
		printf("# unaligned size: dmabuf_attach err=%d\n", err);
		if (err == 0) {
			memset(probe_phys, 0, sizeof(probe_phys));
			err = dmabuf_get_lut(&dmabuf, npages, probe_phys, (int)page);
			printf("# unaligned size: 64K-stride dmabuf_get_lut err=%d\n", err);
			if (err == 0) {
				int matches = 0;
				for (size_t i = 0; i < npages; i++) {
					if (probe_phys[i] == baseline_phys[i]) {
						matches++;
					}
				}
				printf("# unaligned size: %d/%zu phys entries match baseline\n",
				       matches, npages);
			}

			/* 4K-stride probe: npages_4k * host_page == size exactly, so the
			 * iterator boundary case from the 64K-stride attempt is gone. */
			const size_t total_size = aligned_nbytes - 4096;
			const size_t npages_4k = total_size / host_page;
			uint64_t *probe_4k = calloc(npages_4k, sizeof(uint64_t));
			if (probe_4k) {
				err = dmabuf_get_lut(&dmabuf, npages_4k, probe_4k, (int)host_page);
				printf("# unaligned size: 4K-stride dmabuf_get_lut(npages=%zu) "
				       "err=%d\n",
				       npages_4k, err);
				if (err == 0) {
					printf("# unaligned size 4K phys[0,15,16,62]="
					       "0x%" PRIx64 " 0x%" PRIx64 " 0x%" PRIx64
					       " 0x%" PRIx64 "\n",
					       probe_4k[0], probe_4k[15], probe_4k[16],
					       probe_4k[npages_4k - 1]);
					int monotonic = 1;
					for (size_t i = 1; i < npages_4k; i++) {
						if (probe_4k[i] != probe_4k[i - 1] + host_page) {
							printf("# unaligned size: 4K phys "
							       "discontinuity at i=%zu "
							       "(0x%" PRIx64
							       " vs expected 0x%" PRIx64 ")\n",
							       i, probe_4k[i],
							       probe_4k[i - 1] + host_page);
							monotonic = 0;
							break;
						}
					}
					if (monotonic) {
						printf("# unaligned size: 4K-stride phys is "
						       "stride-faithful across all %zu entries\n",
						       npages_4k);
					}
				}
				free(probe_4k);
			}
			dmabuf_detach(&dmabuf);
		} else {
			close(dmabuf_fd);
		}
	}

	/* Probe 3: sub-host-page vaddr (+256). cuMemAlloc itself returns at sub-page
	 * alignment, so this is the case that decides whether
	 * cudamem_mapping_add() could accept its output directly. The dma-buf's
	 * first 4K-stride entry should describe the host-page floor, i.e. match
	 * baseline_phys[0]. */
	cr = cuMemGetHandleForAddressRange(&dmabuf_fd, (CUdeviceptr)((uint8_t *)aligned + 256),
					   aligned_nbytes - 256,
					   CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
	printf("# sub-page vaddr (+256): cuMemGetHandleForAddressRange cr=%d\n", cr);
	if (cr == CUDA_SUCCESS) {
		memset(&dmabuf, 0, sizeof(dmabuf));
		err = dmabuf_attach(dmabuf_fd, &dmabuf);
		printf("# sub-page vaddr: dmabuf_attach err=%d\n", err);
		if (err == 0) {
			const size_t npages_4k = (aligned_nbytes - host_page) / host_page;
			uint64_t *probe_4k = calloc(npages_4k, sizeof(uint64_t));
			if (probe_4k) {
				err = dmabuf_get_lut(&dmabuf, npages_4k, probe_4k, (int)host_page);
				printf("# sub-page vaddr: 4K-stride dmabuf_get_lut(npages=%zu) "
				       "err=%d phys[0]=0x%" PRIx64 "\n",
				       npages_4k, err, err == 0 ? probe_4k[0] : 0);
				if (err == 0) {
					if (probe_4k[0] == baseline_phys[0]) {
						printf("# sub-page vaddr: phys[0] == "
						       "baseline_phys[0] (driver describes "
						       "host-page floor)\n");
					} else {
						printf("# sub-page vaddr: phys[0] (0x%" PRIx64
						       ") != baseline_phys[0] (0x%" PRIx64 ")\n",
						       probe_4k[0], baseline_phys[0]);
					}
					int monotonic = 1;
					for (size_t i = 1; i < npages_4k; i++) {
						if (probe_4k[i] != probe_4k[i - 1] + host_page) {
							printf("# sub-page vaddr: 4K phys "
							       "discontinuity at i=%zu\n",
							       i);
							monotonic = 0;
							break;
						}
					}
					if (monotonic) {
						printf("# sub-page vaddr: 4K-stride phys is "
						       "stride-faithful across all %zu entries\n",
						       npages_4k);
					}
				}
				free(probe_4k);
			}
			dmabuf_detach(&dmabuf);
		} else {
			close(dmabuf_fd);
		}
	}

	/* Probe 4: alignment of raw cuMemAlloc() across sizes. This is what
	 * cudamem_mapping_add() would receive if a caller passed the cuMemAlloc
	 * result directly. Test sizes from sub-page through several pages, and
	 * also a chain of small allocations to see if sub-page packing happens. */
	{
		const size_t sizes[] = {64, 256, 1024, 4096, 65536, 262144};
		for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
			CUdeviceptr raw_probe = 0;
			cr = cuMemAlloc(&raw_probe, sizes[i]);
			if (cr == CUDA_SUCCESS) {
				uintptr_t addr = (uintptr_t)raw_probe;
				printf("# raw cuMemAlloc(%zu): addr=0x%" PRIxPTR
				       " host_page_align=%s dev_page_align=%s "
				       "low12=0x%" PRIxPTR "\n",
				       sizes[i], addr,
				       (addr & (host_page - 1)) == 0 ? "yes" : "no",
				       (addr & (page - 1)) == 0 ? "yes" : "no",
				       addr & ((uintptr_t)host_page - 1));
				cuMemFree(raw_probe);
			} else {
				printf("# raw cuMemAlloc(%zu): cr=%d\n", sizes[i], cr);
			}
		}
		/* Stress sub-page packing: allocate many small blocks back-to-back,
		 * then check if any subsequent block lands sub-page. */
		CUdeviceptr small_probes[16] = {0};
		int packed_subpage = 0;
		for (int i = 0; i < 16; i++) {
			cr = cuMemAlloc(&small_probes[i], 64);
			if (cr != CUDA_SUCCESS) {
				break;
			}
			if (((uintptr_t)small_probes[i] & (host_page - 1)) != 0) {
				printf("# sub-page packing: cuMemAlloc(64)[%d]=0x%" PRIxPTR
				       " low12=0x%" PRIxPTR "\n",
				       i, (uintptr_t)small_probes[i],
				       (uintptr_t)small_probes[i] & ((uintptr_t)host_page - 1));
				packed_subpage = 1;
			}
		}
		if (!packed_subpage) {
			printf("# sub-page packing: every cuMemAlloc(64) returned host-page-aligned "
			       "(driver allocates a whole page per call)\n");
		}
		for (int i = 0; i < 16; i++) {
			if (small_probes[i]) {
				cuMemFree(small_probes[i]);
			}
		}
	}

	cuMemFree(raw);
	return 0;
}

static int
test_add_host_pointer(struct cudamem_mapping_registry *registry, struct cudamem_config *config)
{
	const size_t page = (size_t)config->device_pagesize;
	void *host = NULL;
	int err;

	printf("# test_add_host_pointer\n");

	/* Page-aligned host pointer passes _add's alignment check, but isn't GPU
	 * memory, so cuMemGetHandleForAddressRange must fail. Exercises the
	 * err_free unwind path inside _add. */
	host = aligned_alloc(page, 4 * page);
	if (!host) {
		printf("# FAILED: aligned_alloc; errno(%d)\n", errno);
		return 1;
	}

	err = cudamem_mapping_add(registry, host, 4 * page, NULL);
	if (err == 0) {
		printf("# FAILED: _add(host pointer) unexpectedly succeeded\n");
		cudamem_mapping_clear(registry);
		free(host);
		return 1;
	}
	printf("# _add(host pointer) rejected with err=%d\n", err);

	/* Registry must remain consistent (empty) after the failed _add. */
	if (registry->list != NULL) {
		printf("# FAILED: registry->list non-NULL after failed _add\n");
		free(host);
		return 1;
	}
	if (registry->va_base != UINT64_MAX) {
		printf("# FAILED: va_base mutated by failed _add\n");
		free(host);
		return 1;
	}

	free(host);
	return 0;
}

int
main(void)
{
	struct cudamem_config config = {0};
	struct cudamem_mapping_registry registry = {0};
	CUdevice cu_dev;
	CUcontext cu_ctx;
	int err;

	err = cuInit(0);
	if (err) {
		printf("# FAILED: cuInit(); err(%d)\n", err);
		return err;
	}

	err = cuDeviceGet(&cu_dev, 0);
	if (err) {
		printf("# FAILED: cuDeviceGet(); err(%d)\n", err);
		return err;
	}

	err = cuCtxCreate(&cu_ctx, 0, cu_dev);
	if (err) {
		printf("# FAILED: cuCtxCreate(); err(%d)\n", err);
		return err;
	}

	err = cudamem_config_init(&config, 0);
	if (err) {
		printf("# FAILED: cudamem_config_init(); err(%d)\n", err);
		goto exit_ctx;
	}

	err = cudamem_mapping_registry_init(&registry, &config);
	if (err) {
		printf("# FAILED: cudamem_mapping_registry_init(); err(%d)\n", err);
		goto exit_ctx;
	}

	err = test_add_remove_clear(&registry, &config);
	if (err) {
		goto exit;
	}
	err = test_unaligned_lookup(&registry, &config);
	if (err) {
		goto exit;
	}
	err = test_misalignment(&registry, &config);
	if (err) {
		goto exit;
	}
	err = test_lookup_unmapped(&registry, &config);
	if (err) {
		goto exit;
	}
	err = test_multiple_mappings(&registry, &config);
	if (err) {
		goto exit;
	}
	err = test_va_base_shift(&registry, &config);
	if (err) {
		goto exit;
	}
	err = test_cumemgethandle_unaligned(&config);
	if (err) {
		goto exit;
	}
	err = test_add_host_pointer(&registry, &config);
	if (err) {
		goto exit;
	}

	printf("SUCCES: all cudamem_mapping tests passed\n");

exit:
	cudamem_mapping_registry_term(&registry);
exit_ctx:
	cuCtxDestroy(cu_ctx);
	return err;
}
