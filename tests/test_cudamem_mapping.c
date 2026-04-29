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
			printf("# sub-page packing: every cuMemAlloc(64) returned "
			       "host-page-aligned "
			       "(driver allocates a whole page per call)\n");
		}
		for (int i = 0; i < 16; i++) {
			if (small_probes[i]) {
				cuMemFree(small_probes[i]);
			}
		}
	}

	/* Probe 5: floored sub-page slab allocation. Take a sub-page-aligned
	 * cuMemAlloc(64) result, floor to host page, and see if
	 * cuMemGetHandleForAddressRange + dmabuf_get_lut work on the floored
	 * range. Also probe what happens after the leading aligned allocation
	 * (which sits at the floor) is freed. */
	{
		CUdeviceptr aligned_anchor = 0; /* sits at the host-page boundary */
		CUdeviceptr subpage_alloc = 0;  /* sub-page suballocation in the same page */
		cr = cuMemAlloc(&aligned_anchor, 64);
		if (cr != CUDA_SUCCESS) {
			printf("# floored slab: cuMemAlloc(anchor) cr=%d\n", cr);
			goto floored_done;
		}
		cr = cuMemAlloc(&subpage_alloc, 64);
		if (cr != CUDA_SUCCESS) {
			printf("# floored slab: cuMemAlloc(subpage) cr=%d\n", cr);
			cuMemFree(aligned_anchor);
			goto floored_done;
		}
		const uintptr_t anchor_addr = (uintptr_t)aligned_anchor;
		const uintptr_t sub_addr = (uintptr_t)subpage_alloc;
		const uintptr_t floored = sub_addr & ~((uintptr_t)host_page - 1);
		printf("# floored slab: anchor=0x%" PRIxPTR " subpage=0x%" PRIxPTR
		       " (low12=0x%" PRIxPTR ") floored=0x%" PRIxPTR "\n",
		       anchor_addr, sub_addr, sub_addr & (host_page - 1), floored);

		if ((sub_addr & (host_page - 1)) == 0 || floored != anchor_addr) {
			printf("# floored slab: skipping; subpage isn't sub-page-aligned within "
			       "anchor's page (slab packing not as expected)\n");
		} else {
			/* Step 1: cuMemGetHandleForAddressRange on the floored range while
			 * the anchor is still live. */
			cr = cuMemGetHandleForAddressRange(&dmabuf_fd, (CUdeviceptr)floored,
							   host_page,
							   CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
			printf("# floored slab (anchor live): cuMemGetHandleForAddressRange "
			       "cr=%d\n",
			       cr);
			if (cr == CUDA_SUCCESS) {
				memset(&dmabuf, 0, sizeof(dmabuf));
				err = dmabuf_attach(dmabuf_fd, &dmabuf);
				printf("# floored slab (anchor live): dmabuf_attach err=%d\n",
				       err);
				if (err == 0) {
					uint64_t phys = 0;
					err = dmabuf_get_lut(&dmabuf, 1, &phys, (int)host_page);
					printf("# floored slab (anchor live): dmabuf_get_lut "
					       "err=%d "
					       "phys=0x%" PRIx64 "\n",
					       err, phys);
					dmabuf_detach(&dmabuf);
				} else {
					close(dmabuf_fd);
				}
			}

			/* Step 2: free the anchor; subpage_alloc is still live in the same
			 * physical page. Repeat the probe. */
			cuMemFree(aligned_anchor);
			aligned_anchor = 0;
			cr = cuMemGetHandleForAddressRange(&dmabuf_fd, (CUdeviceptr)floored,
							   host_page,
							   CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
			printf("# floored slab (anchor freed): cuMemGetHandleForAddressRange "
			       "cr=%d\n",
			       cr);
			if (cr == CUDA_SUCCESS) {
				memset(&dmabuf, 0, sizeof(dmabuf));
				err = dmabuf_attach(dmabuf_fd, &dmabuf);
				printf("# floored slab (anchor freed): dmabuf_attach err=%d\n",
				       err);
				if (err == 0) {
					uint64_t phys = 0;
					err = dmabuf_get_lut(&dmabuf, 1, &phys, (int)host_page);
					printf("# floored slab (anchor freed): dmabuf_get_lut "
					       "err=%d "
					       "phys=0x%" PRIx64 "\n",
					       err, phys);
					dmabuf_detach(&dmabuf);
				} else {
					close(dmabuf_fd);
				}
			}
		}

		if (aligned_anchor) {
			cuMemFree(aligned_anchor);
		}
		cuMemFree(subpage_alloc);
	}
floored_done:

	/* Probe 6: phys consistency across overlapping cuMemGetHandleForAddressRange
	 * calls. The floor-and-register-with-refcount design relies on this:
	 * separate _add calls whose floored ranges overlap on a host page must
	 * agree on the phys for that page (otherwise the rc-incremented LUT slot
	 * would have to verify, not just bump). Drive two sub-page slab
	 * allocations into the same host page, register the floored host-page
	 * range from each separately, and compare. */
	{
		CUdeviceptr alloc_a = 0;
		CUdeviceptr alloc_b = 0;
		cr = cuMemAlloc(&alloc_a, 64);
		if (cr != CUDA_SUCCESS) {
			printf("# phys consistency: cuMemAlloc(a) cr=%d\n", cr);
			goto consistency_done;
		}
		cr = cuMemAlloc(&alloc_b, 64);
		if (cr != CUDA_SUCCESS) {
			printf("# phys consistency: cuMemAlloc(b) cr=%d\n", cr);
			cuMemFree(alloc_a);
			goto consistency_done;
		}
		const uintptr_t a_addr = (uintptr_t)alloc_a;
		const uintptr_t b_addr = (uintptr_t)alloc_b;
		const uintptr_t a_floor = a_addr & ~((uintptr_t)host_page - 1);
		const uintptr_t b_floor = b_addr & ~((uintptr_t)host_page - 1);
		printf("# phys consistency: a=0x%" PRIxPTR " b=0x%" PRIxPTR
		       " a_floor=0x%" PRIxPTR " b_floor=0x%" PRIxPTR "\n",
		       a_addr, b_addr, a_floor, b_floor);

		if (a_floor != b_floor) {
			printf("# phys consistency: skipping; slab packing did not place "
			       "both allocs in the same host page\n");
		} else {
			uint64_t phys_a = 0, phys_b = 0;
			struct dmabuf db_a = {0}, db_b = {0};
			int fd_a = -1, fd_b = -1;
			int got_a = 0, got_b = 0;

			cr = cuMemGetHandleForAddressRange(
				&fd_a, (CUdeviceptr)a_floor, host_page,
				CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
			printf("# phys consistency: cuMemGetHandleForAddressRange(a_floor) "
			       "cr=%d\n",
			       cr);
			if (cr == CUDA_SUCCESS) {
				err = dmabuf_attach(fd_a, &db_a);
				if (err == 0) {
					err = dmabuf_get_lut(&db_a, 1, &phys_a, (int)host_page);
					printf("# phys consistency: a_floor dmabuf_get_lut err=%d "
					       "phys=0x%" PRIx64 "\n",
					       err, phys_a);
					got_a = (err == 0);
				} else {
					close(fd_a);
				}
			}

			cr = cuMemGetHandleForAddressRange(
				&fd_b, (CUdeviceptr)b_floor, host_page,
				CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
			printf("# phys consistency: cuMemGetHandleForAddressRange(b_floor) "
			       "cr=%d\n",
			       cr);
			if (cr == CUDA_SUCCESS) {
				err = dmabuf_attach(fd_b, &db_b);
				if (err == 0) {
					err = dmabuf_get_lut(&db_b, 1, &phys_b, (int)host_page);
					printf("# phys consistency: b_floor dmabuf_get_lut err=%d "
					       "phys=0x%" PRIx64 "\n",
					       err, phys_b);
					got_b = (err == 0);
				} else {
					close(fd_b);
				}
			}

			if (got_a && got_b) {
				if (phys_a == phys_b && phys_a != 0) {
					printf("# phys consistency: PASS - separate dma-bufs "
					       "agree on phys (0x%" PRIx64 ") for shared host "
					       "page\n",
					       phys_a);
				} else {
					printf("# phys consistency: FAIL - phys_a=0x%" PRIx64
					       " phys_b=0x%" PRIx64 "\n",
					       phys_a, phys_b);
				}

				/* Also probe rc-low-bits assumption: phys must have low
				 * pagesize_shift bits zero so we can pack rc into them. */
				if ((phys_a & (host_page - 1)) == 0) {
					printf("# phys consistency: low %d bits of phys are "
					       "zero (rc-bits available)\n",
					       config->pagesize_shift);
				} else {
					printf("# phys consistency: WARN - phys 0x%" PRIx64
					       " is not host-page aligned; rc cannot live in "
					       "low bits\n",
					       phys_a);
				}
			}

			if (got_a) {
				dmabuf_detach(&db_a);
			}
			if (got_b) {
				dmabuf_detach(&db_b);
			}
		}

		cuMemFree(alloc_a);
		cuMemFree(alloc_b);
	}
consistency_done:

	/* Probe 7: phys stability of repeated identical calls at device_pagesize
	 * stride. Probe 6 showed two calls on the same host-page range return
	 * different phys. This checks whether device-page granularity changes
	 * that — does the kernel cache BAR1 mappings per device-page request, so
	 * back-to-back identical calls reuse the same IOVA? Use the test's
	 * existing device-page-aligned `aligned` so the input is unambiguously a
	 * valid CUDA VA range. */
	{
		int fd1 = -1, fd2 = -1;
		struct dmabuf db1 = {0}, db2 = {0};
		uint64_t phys1 = 0, phys2 = 0;
		int got1 = 0, got2 = 0;

		cr = cuMemGetHandleForAddressRange(&fd1, (CUdeviceptr)aligned, page,
						   CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
		printf("# device-page identical-call: get(call 1) cr=%d\n", cr);
		if (cr == CUDA_SUCCESS) {
			err = dmabuf_attach(fd1, &db1);
			if (err == 0) {
				err = dmabuf_get_lut(&db1, 1, &phys1, (int)page);
				got1 = (err == 0);
			} else {
				close(fd1);
			}
		}

		cr = cuMemGetHandleForAddressRange(&fd2, (CUdeviceptr)aligned, page,
						   CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
		printf("# device-page identical-call: get(call 2) cr=%d\n", cr);
		if (cr == CUDA_SUCCESS) {
			err = dmabuf_attach(fd2, &db2);
			if (err == 0) {
				err = dmabuf_get_lut(&db2, 1, &phys2, (int)page);
				got2 = (err == 0);
			} else {
				close(fd2);
			}
		}

		if (got1 && got2) {
			printf("# device-page identical-call: phys1=0x%" PRIx64
			       " phys2=0x%" PRIx64 " — %s\n",
			       phys1, phys2,
			       phys1 == phys2 ? "STABLE (rc design viable)"
					      : "DIFFER (BAR1 IOVA fresh per call)");
		}

		if (got1) {
			dmabuf_detach(&db1);
		}
		if (got2) {
			dmabuf_detach(&db2);
		}
	}

	/* Probe 8: BAR1 IOVA reservation quantum vs request size. Probes 6 and 7
	 * showed adjacent attachments differ by 0x200000 (2 MiB) for sub-2-MiB
	 * requests. Test whether the per-attachment IOVA chunk stays at 2 MiB
	 * for larger requests too, or scales with size. Allocate a backing
	 * buffer, then issue two simultaneous attachments on adjacent halves of
	 * varying sizes and observe phys[0] delta. */
	{
		const size_t backing_size = (size_t)16 * 1024 * 1024;
		const size_t test_sizes[] = {
			(size_t)4096,
			(size_t)65536,
			(size_t)256 * 1024,
			(size_t)1024 * 1024,
			(size_t)2 * 1024 * 1024,
			(size_t)4 * 1024 * 1024,
		};
		CUdeviceptr backing_raw = 0;
		void *backing_aligned = NULL;
		size_t backing_aligned_nbytes = 0;

		err = cudamem_alloc_aligned(config, backing_size, &backing_raw, &backing_aligned,
					    &backing_aligned_nbytes);
		if (err) {
			printf("# probe 8: cudamem_alloc_aligned failed; err=%d\n", err);
			goto probe8_done;
		}

		for (size_t i = 0; i < sizeof(test_sizes) / sizeof(test_sizes[0]); i++) {
			const size_t sz = test_sizes[i];
			if (2 * sz > backing_aligned_nbytes) {
				printf("# probe 8 (sz=%9zu): skip; backing too small\n", sz);
				continue;
			}
			const size_t nphys = sz / host_page;
			uint64_t *lut1 = calloc(nphys, sizeof(uint64_t));
			uint64_t *lut2 = calloc(nphys, sizeof(uint64_t));
			int fd1 = -1, fd2 = -1;
			struct dmabuf db1 = {0}, db2 = {0};
			int got1 = 0, got2 = 0;

			if (!lut1 || !lut2) {
				free(lut1);
				free(lut2);
				continue;
			}

			cr = cuMemGetHandleForAddressRange(&fd1, (CUdeviceptr)backing_aligned, sz,
							   CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
							   0);
			if (cr == CUDA_SUCCESS) {
				err = dmabuf_attach(fd1, &db1);
				if (err == 0) {
					err = dmabuf_get_lut(&db1, nphys, lut1, (int)host_page);
					got1 = (err == 0);
				} else {
					close(fd1);
				}
			}

			cr = cuMemGetHandleForAddressRange(
				&fd2, (CUdeviceptr)((uint8_t *)backing_aligned + sz), sz,
				CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
			if (cr == CUDA_SUCCESS) {
				err = dmabuf_attach(fd2, &db2);
				if (err == 0) {
					err = dmabuf_get_lut(&db2, nphys, lut2, (int)host_page);
					got2 = (err == 0);
				} else {
					close(fd2);
				}
			}

			if (got1 && got2) {
				int64_t delta = (int64_t)lut2[0] - (int64_t)lut1[0];
				uint64_t adelta = (uint64_t)(delta < 0 ? -delta : delta);
				printf("# probe 8 (sz=%9zu): phys1=0x%" PRIx64
				       " phys2=0x%" PRIx64 " |delta|=0x%" PRIx64
				       " (%llu bytes, %llux request)\n",
				       sz, lut1[0], lut2[0], adelta,
				       (unsigned long long)adelta,
				       (unsigned long long)(adelta / sz));
			}

			if (got1) {
				dmabuf_detach(&db1);
			}
			if (got2) {
				dmabuf_detach(&db2);
			}
			free(lut1);
			free(lut2);
		}

		cuMemFree(backing_raw);
	}
probe8_done:

	/* Probe 9: does cuMemGetHandleForAddressRange accept a 2 MiB-floored
	 * range when only a sub-region is explicitly cuMemAlloc'd? If yes, the
	 * dmabuf-cache-keyed-by-2MiB-chunk design works straightforwardly. If
	 * no, the design needs an exact-range fallback. Test with a single
	 * small cuMemAlloc and try requesting the 2 MiB chunk that contains
	 * it. */
	{
		const size_t chunk = (size_t)2 * 1024 * 1024;
		CUdeviceptr small = 0;
		cr = cuMemAlloc(&small, 64);
		if (cr != CUDA_SUCCESS) {
			printf("# probe 9: cuMemAlloc(64) cr=%d\n", cr);
			goto probe9_done;
		}
		const uintptr_t small_addr = (uintptr_t)small;
		const uintptr_t chunk_floor = small_addr & ~(uintptr_t)(chunk - 1);
		printf("# probe 9: small=0x%" PRIxPTR " chunk_floor=0x%" PRIxPTR
		       " (offset within chunk=0x%" PRIxPTR ")\n",
		       small_addr, chunk_floor, small_addr - chunk_floor);

		int probe_fd = -1;
		cr = cuMemGetHandleForAddressRange(&probe_fd, (CUdeviceptr)chunk_floor, chunk,
						   CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
		printf("# probe 9: cuMemGetHandleForAddressRange(chunk_floor, 2 MiB) cr=%d\n",
		       cr);
		if (cr == CUDA_SUCCESS) {
			struct dmabuf db = {0};
			err = dmabuf_attach(probe_fd, &db);
			printf("# probe 9: dmabuf_attach err=%d\n", err);
			if (err == 0) {
				size_t nphys = chunk / host_page;
				uint64_t *lut = calloc(nphys, sizeof(uint64_t));
				if (lut) {
					err = dmabuf_get_lut(&db, nphys, lut, (int)host_page);
					printf("# probe 9: dmabuf_get_lut(nphys=%zu) err=%d "
					       "phys[0]=0x%" PRIx64 " phys[last]=0x%" PRIx64 "\n",
					       nphys, err, err == 0 ? lut[0] : 0,
					       err == 0 ? lut[nphys - 1] : 0);
					free(lut);
				}
				dmabuf_detach(&db);
			} else {
				close(probe_fd);
			}
			printf("# probe 9: PASS - 2 MiB-floored request accepted; "
			       "cache-keyed-by-chunk design viable\n");
		} else {
			printf("# probe 9: FAIL - 2 MiB-floored request rejected; "
			       "need exact-range fallback\n");
		}

		cuMemFree(small);
	}
probe9_done:

	/* Probe 10: chunk-size sweep. For each candidate chunk, run two
	 * sub-probes:
	 *   (a) tiny: cuMemAlloc(64), floor to chunk, export chunk.
	 *       answers "what's the slab/reservation granularity of cuMemAlloc?"
	 *   (b) full: cuMemAlloc(chunk + chunk), floor to chunk, export chunk.
	 *       answers "what range sizes will the driver export when fully
	 *       backed?"
	 * The candidate alignments span sub-2MiB (to test if smaller floors
	 * work), 2 MiB (the known-good baseline), and supra-2MiB (to test if
	 * the export can cover multi-slab spans). */
	{
		static const size_t chunks[] = {
			(size_t)64 * 1024,
			(size_t)256 * 1024,
			(size_t)1024 * 1024,
			(size_t)2 * 1024 * 1024,
			(size_t)4 * 1024 * 1024,
			(size_t)8 * 1024 * 1024,
			(size_t)16 * 1024 * 1024,
			(size_t)64 * 1024 * 1024,
		};
		const size_t ncases = sizeof(chunks) / sizeof(chunks[0]);

		for (size_t ci = 0; ci < ncases; ++ci) {
			const size_t chunk = chunks[ci];

			/* (a) tiny alloc */
			{
				CUdeviceptr small = 0;
				cr = cuMemAlloc(&small, 64);
				if (cr != CUDA_SUCCESS) {
					printf("# probe 10 (chunk=%9zu, tiny): cuMemAlloc cr=%d\n",
					       chunk, cr);
				} else {
					const uintptr_t addr = (uintptr_t)small;
					const uintptr_t floor = addr & ~(uintptr_t)(chunk - 1);
					int fd = -1;
					cr = cuMemGetHandleForAddressRange(
						&fd, (CUdeviceptr)floor, chunk,
						CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
					printf("# probe 10 (chunk=%9zu, tiny): addr=0x%" PRIxPTR
					       " floor=0x%" PRIxPTR " offset=0x%" PRIxPTR
					       " cr=%d %s\n",
					       chunk, addr, floor, addr - floor, cr,
					       cr == CUDA_SUCCESS ? "PASS" : "FAIL");
					if (cr == CUDA_SUCCESS) {
						close(fd);
					}
					cuMemFree(small);
				}
			}

			/* (b) full alloc that definitely spans the chunk */
			{
				CUdeviceptr big = 0;
				const size_t bigsz = chunk * 2;
				cr = cuMemAlloc(&big, bigsz);
				if (cr != CUDA_SUCCESS) {
					printf("# probe 10 (chunk=%9zu, full): cuMemAlloc(%zu) cr=%d\n",
					       chunk, bigsz, cr);
				} else {
					const uintptr_t addr = (uintptr_t)big;
					/* Round the addr up to chunk alignment so the
					 * full chunk is guaranteed inside the alloc. */
					const uintptr_t aligned =
						(addr + chunk - 1) & ~(uintptr_t)(chunk - 1);
					int fd = -1;
					cr = cuMemGetHandleForAddressRange(
						&fd, (CUdeviceptr)aligned, chunk,
						CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
					printf("# probe 10 (chunk=%9zu, full): addr=0x%" PRIxPTR
					       " aligned=0x%" PRIxPTR " cr=%d %s\n",
					       chunk, addr, aligned, cr,
					       cr == CUDA_SUCCESS ? "PASS" : "FAIL");
					if (cr == CUDA_SUCCESS) {
						close(fd);
					}
					cuMemFree(big);
				}
			}
		}
	}

	/* Probe 11: can we query the slab granularity?
	 *
	 * (a) cuMemGetAllocationGranularity is documented as VMM-only, but
	 *     report what it returns for a pinned-device-mem prop on this
	 *     device, both MINIMUM and RECOMMENDED. If it matches the
	 *     empirically-observed 2 MiB, we have a portable query.
	 * (b) Runtime probe: tiny cuMemAlloc, then exponentially increase the
	 *     chunk request until cuMemGetHandleForAddressRange fails; the
	 *     last successful size is the slab granularity for cuMemAlloc on
	 *     this system. */
	{
		CUdevice dev;
		cr = cuCtxGetDevice(&dev);
		if (cr != CUDA_SUCCESS) {
			printf("# probe 11: cuCtxGetDevice cr=%d\n", cr);
			goto probe11_done;
		}

		CUmemAllocationProp prop = {0};
		prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
		prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		prop.location.id = (int)dev;

		size_t gran_min = 0;
		size_t gran_rec = 0;
		cr = cuMemGetAllocationGranularity(&gran_min, &prop,
						   CU_MEM_ALLOC_GRANULARITY_MINIMUM);
		printf("# probe 11: cuMemGetAllocationGranularity(MINIMUM) cr=%d size=%zu\n",
		       cr, gran_min);
		cr = cuMemGetAllocationGranularity(&gran_rec, &prop,
						   CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
		printf("# probe 11: cuMemGetAllocationGranularity(RECOMMENDED) cr=%d size=%zu\n",
		       cr, gran_rec);

		/* Runtime probe: find the max floored chunk size that succeeds
		 * for a single tiny cuMemAlloc. */
		CUdeviceptr small = 0;
		cr = cuMemAlloc(&small, 64);
		if (cr != CUDA_SUCCESS) {
			printf("# probe 11 (runtime): cuMemAlloc cr=%d\n", cr);
			goto probe11_done;
		}
		size_t max_ok = 0;
		for (size_t sz = (size_t)4 * 1024; sz <= (size_t)128 * 1024 * 1024; sz <<= 1) {
			const uintptr_t floor = (uintptr_t)small & ~(uintptr_t)(sz - 1);
			int fd = -1;
			CUresult cr2 = cuMemGetHandleForAddressRange(
				&fd, (CUdeviceptr)floor, sz,
				CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
			if (cr2 == CUDA_SUCCESS) {
				max_ok = sz;
				close(fd);
			} else {
				printf("# probe 11 (runtime): first failure at chunk=%zu (cr=%d)\n",
				       sz, cr2);
				break;
			}
		}
		printf("# probe 11 (runtime): max successful floored chunk = %zu (= %zu MiB)\n",
		       max_ok, max_ok >> 20);
		cuMemFree(small);
	}
probe11_done:

	/* Probe 12: phys contiguity within a slab, across many slabs.
	 *
	 * The slab-LUT design assumes that within a single 2 MiB chunk
	 * exported via cuMemGetHandleForAddressRange, the host-page phys
	 * entries are contiguous (phys[i] == phys[0] + i * pagesize). This
	 * follows from the BAR1 page table mapping at slab granularity, but
	 * verify empirically across many chunks to confirm the property is
	 * not occasionally violated (e.g., by fragmented BAR1 PT walks).
	 *
	 * Allocate a multi-slab region, walk slab-aligned windows within it,
	 * export each, and check contiguity of the LUT. */
	{
		const size_t slab = (size_t)2 * 1024 * 1024;
		const size_t nslabs_test = 32;
		const size_t big_sz = slab * nslabs_test;
		CUdeviceptr big = 0;
		cr = cuMemAlloc(&big, big_sz);
		if (cr != CUDA_SUCCESS) {
			printf("# probe 12: cuMemAlloc(%zu) cr=%d\n", big_sz, cr);
			goto probe12_done;
		}

		const uintptr_t big_addr = (uintptr_t)big;
		const uintptr_t base_aligned = (big_addr + slab - 1) & ~(uintptr_t)(slab - 1);
		const size_t lut_n = slab / host_page;
		uint64_t *lut = calloc(lut_n, sizeof(uint64_t));
		if (!lut) {
			printf("# probe 12: calloc failed\n");
			cuMemFree(big);
			goto probe12_done;
		}

		size_t chunks_checked = 0;
		size_t chunks_contig = 0;
		size_t first_violation_at = (size_t)-1;
		uint64_t first_violation_phys[2] = {0, 0};
		size_t first_violation_idx = 0;

		const size_t walkable = (big_sz - (base_aligned - big_addr)) / slab;
		for (size_t s = 0; s < walkable; ++s) {
			const uintptr_t chunk_va = base_aligned + s * slab;
			int fd = -1;
			cr = cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)chunk_va, slab,
							   CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
			if (cr != CUDA_SUCCESS) {
				continue;
			}
			struct dmabuf db = {0};
			err = dmabuf_attach(fd, &db);
			if (err) {
				close(fd);
				continue;
			}
			err = dmabuf_get_lut(&db, lut_n, lut, (int)host_page);
			if (err == 0) {
				chunks_checked++;
				int contig = 1;
				for (size_t i = 1; i < lut_n; ++i) {
					if (lut[i] != lut[0] + i * host_page) {
						contig = 0;
						if (first_violation_at == (size_t)-1) {
							first_violation_at = s;
							first_violation_idx = i;
							first_violation_phys[0] = lut[0];
							first_violation_phys[1] = lut[i];
						}
						break;
					}
				}
				if (contig) {
					chunks_contig++;
				}
			}
			dmabuf_detach(&db);
		}

		printf("# probe 12: chunks_checked=%zu chunks_contig=%zu\n",
		       chunks_checked, chunks_contig);
		if (chunks_checked == chunks_contig && chunks_checked > 0) {
			printf("# probe 12: PASS - all chunks contiguous; slab-LUT design "
			       "viable\n");
		} else if (chunks_checked == 0) {
			printf("# probe 12: SKIP - no chunks could be checked\n");
		} else {
			printf("# probe 12: FAIL - first violation at chunk %zu, page %zu: "
			       "phys[0]=0x%" PRIx64 " phys[%zu]=0x%" PRIx64 "\n",
			       first_violation_at, first_violation_idx,
			       first_violation_phys[0], first_violation_idx,
			       first_violation_phys[1]);
		}

		free(lut);
		cuMemFree(big);
	}
probe12_done:

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
