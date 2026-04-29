# cudamem_mapping probe findings

Exploratory probes added to `tests/test_cudamem_mapping.c` to characterise
behaviour of `cuMemAlloc`, `cuMemGetHandleForAddressRange`, and the dma-buf
export path on NVIDIA discrete GPUs. The goal: inform the design of
`cudamem_mapping_registry` for framework-allocated CUDA buffers, where
allocations may be slab-packed below host-page alignment.

Hardware: NVIDIA RTX 2000 Ada (`swissknife`, CUDA 12.8, kernel 6.8.12-dmabuf,
`iommu=off`).

## Summary of conclusions

- `cuMemAlloc` reserves VA in 2 MiB slabs. Sub-page packing within a slab is
  visible to userspace (`cuMemAlloc(64)` returns sub-page addresses).
- The slab granularity is queryable via `cuMemGetAllocationGranularity` with
  a pinned-device-mem prop; despite the API being documented as VMM-only, it
  returns the same 2 MiB value as the empirical cuMemAlloc slab.
- `cuMemGetHandleForAddressRange` accepts any `device_pagesize` (64 KiB)
  multiple as the floor and size, provided the entire requested range is
  backed by `cuMemAlloc`. With a single tiny allocation, the maximum
  exportable range is exactly one slab (2 MiB). With a multi-slab allocation,
  exports up to 64 MiB succeed (no upper bound found).
- Each `cuMemGetHandleForAddressRange` call is a fresh dma-buf export with a
  fresh BAR1 IOVA window. Two calls on identical VA ranges return different
  phys. The IOVA reservation quantum is 2 MiB (matching the BAR1
  large-page size), independent of request size.
- Implication for the registry: an rc-share-phys design over the user's VA is
  unsound, since separate exports produce different phys for the same VA.
  A viable design caches dmabufs at 2 MiB chunk granularity; reuse within a
  cached dmabuf preserves phys, restoring the rc-share-phys invariant at the
  cache layer.

## Probe-by-probe results

### Probe 5: floored sub-page slab allocation

Two `cuMemAlloc(64)` calls land in the same host page; the first is
host-page aligned (the anchor), the second is at offset 0x200 within that
page. Floor the second to host-page granularity and call
`cuMemGetHandleForAddressRange` on the floored range. Repeat after freeing
the anchor.

```
floored slab: anchor=0x...441000 subpage=0x...441200 (low12=0x200) floored=0x...441000
floored slab (anchor live):  cr=0  phys=0x7800441000
floored slab (anchor freed): cr=0  phys=0x7800441000
```

Conclusion: the floored host-page export succeeds in both states, and the
phys returned is identical. The driver does not invalidate VA reservations
of an enclosing slab when an inner allocation is freed.

### Probe 6: phys consistency across overlapping floored exports

Two sub-page `cuMemAlloc(64)` allocations packed into the same host page.
Export the floored host-page range from each separately, compare phys.

```
phys consistency: a=0x...441000 b=0x...441200 a_floor=0x...441000 b_floor=0x...441000
phys consistency: a_floor phys=0x7800441000
phys consistency: b_floor phys=0x7800641000
phys consistency: FAIL - phys_a=0x7800441000 phys_b=0x7800641000
```

Conclusion: distinct exports of the same VA produce distinct phys, separated
by 2 MiB (one BAR1 IOVA window). An rc-share-phys design keyed on the
user's host-page VA is unsound: the second `_add` cannot inherit the first's
phys.

### Probe 7: phys stability under identical back-to-back calls

Two `cuMemGetHandleForAddressRange` calls with identical arguments
(device-page aligned VA, `device_pagesize` length).

```
device-page identical-call: phys1=0x7800400000 phys2=0x7800600000
                             — DIFFER (BAR1 IOVA fresh per call)
```

Conclusion: the kernel does not deduplicate identical export requests. Each
call burns a fresh IOVA window. Reinforces probe 6.

### Probe 8: BAR1 IOVA reservation quantum

For a range of request sizes, allocate two adjacent CUDA blocks, export each,
compare the phys delta.

```
sz=     4096:  |delta|=0x201000  (2 MiB + 4 KiB)
sz=    65536:  |delta|=0x210000  (2 MiB + 64 KiB)
sz=   262144:  |delta|=0x240000  (2 MiB + 256 KiB)
sz=  1048576:  |delta|=0x300000  (2 MiB + 1 MiB)
sz=  2097152:  |delta|=0x200000  (exactly 2 MiB)
sz=  4194304:  |delta|=0x400000  (exactly 4 MiB)
```

Conclusion: each export burns `ceil(sz, 2 MiB)` IOVA. The 2 MiB quantum
matches the BAR1 page-table large-page size. Sub-2-MiB exports do not save
IOVA.

### Probe 9: 2 MiB-floored export with sub-region allocation

Single tiny `cuMemAlloc(64)`. Floor to 2 MiB, request a 2 MiB export.

```
probe 9: small=0x...441000 chunk_floor=0x...400000 (offset within chunk=0x41000)
probe 9: cuMemGetHandleForAddressRange(chunk_floor, 2 MiB) cr=0
probe 9: dmabuf_get_lut(nphys=512) phys[0]=0x7800400000 phys[last]=0x78005ff000
```

Conclusion: the export succeeds and the full 512-page LUT is populated, even
though only 64 bytes are explicitly allocated. The slab is fully backed.
This is the foundation of the cache-keyed-by-2-MiB-chunk design.

### Probe 10: chunk-size sweep

Two scenarios per chunk size:

- (a) Tiny: `cuMemAlloc(64)`, floor user VA to chunk, request `chunk` bytes.
- (b) Full: `cuMemAlloc(2 * chunk)`, request a chunk-aligned `chunk` bytes
  inside.

| Chunk    | Tiny  | Full |
| -------- | ----- | ---- |
| 64 KiB   | PASS  | PASS |
| 256 KiB  | PASS  | PASS |
| 1 MiB    | PASS  | PASS |
| 2 MiB    | PASS  | PASS |
| 4 MiB    | FAIL  | PASS |
| 8 MiB    | FAIL  | PASS |
| 16 MiB   | FAIL  | PASS |
| 64 MiB   | FAIL  | PASS |

Conclusions:

- A single tiny allocation reserves exactly 2 MiB of VA. Floored requests of
  4 MiB or larger fail because the adjacent slab is unbacked.
- The minimum acceptable floor is `device_pagesize` (64 KiB); sub-2-MiB
  floors all succeed.
- The maximum acceptable export size is bounded only by the size of the
  contiguously-backed VA (no kernel-imposed upper limit observed up to
  64 MiB).

### Probe 11: querying the slab granularity

Two paths: the documented API, and a runtime probe.

```
cuMemGetAllocationGranularity(MINIMUM)     = 2097152
cuMemGetAllocationGranularity(RECOMMENDED) = 2097152
runtime probe (max successful floor)       = 2097152
```

Conclusion: `cuMemGetAllocationGranularity` returns the same 2 MiB value as
the empirical cuMemAlloc slab boundary, despite being documented as VMM-only.
This gives a portable query for the cache chunk size, which is needed
because the value is per-device by API contract (and varies in practice
across GPU architectures).

### Probe 12: phys contiguity within a slab

Allocate 64 MiB (32 slabs). For each slab-aligned window, export it as a
separate dmabuf, fetch the 512-page LUT, and verify
`lut[i] == lut[0] + i * pagesize` for all `i`.

```
probe 12: chunks_checked=32 chunks_contig=32
probe 12: PASS - all chunks contiguous; slab-LUT design viable
```

Conclusion: each chunk export receives a single contiguous BAR1 IOVA
window. This is consistent with BAR1 large-page mapping at slab granularity.
A registry that stores one phys base per chunk and computes per-page phys as
`base + (va & (slab - 1))` is sound on this hardware.

## Design implications

The cache should:

- Key on slab-aligned VA chunks. Use
  `cuMemGetAllocationGranularity(prop=PINNED, MINIMUM)` to obtain the slab
  size at config-init time, store on `cudamem_config`. Do not hardcode 2 MiB.
- Manage one dma-buf per chunk, with a refcount. `_add` rounds out to chunk
  boundaries, ref-bumps existing entries, populates new entries via
  `cuMemGetHandleForAddressRange(chunk_floor, chunk)`.
- Store one `phys_base` per chunk (probe 12 verified contiguity); resolve
  `virt_to_phys` as `chunk->phys_base + (va & (slab - 1))`. The host-page
  LUT used in the existing registry can be eliminated, reducing memory by
  the slab/pagesize ratio (512x for 4 KiB pages and 2 MiB slabs).
- At chunk init, populate `phys_base` by calling `dmabuf_get_lut` for the
  full slab and asserting contiguity, then storing `lut[0]`. Refuse with
  `-EOPNOTSUPP` on violation; this guards against hardware where the BAR1
  large-page assumption does not hold.
- Free dmabufs only when the chunk rc reaches zero. Until then, repeated
  `_add`/`_remove` cycles touching the same chunk pay zero CUDA cost and
  preserve phys identity.
