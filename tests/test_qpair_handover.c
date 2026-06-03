// SPDX-License-Identifier: BSD-3-Clause

/**
 * Cross-process NVMe I/O qpair handout
 * ====================================
 *
 * Two processes, one physical NVMe controller:
 *
 *   owner   opens the controller (resets it, owns the admin queue), creates an
 *           I/O qpair whose SQ/CQ rings live in a shared (memfd) hugepage, and
 *           writes a descriptor of that qpair to a file. It then stays alive so
 *           the shared region and controller remain up.
 *
 *   client  reads the descriptor, maps the shared region (for the rings) and
 *           BAR0 (for the doorbells), builds a qpair with nvme_qpair_import(),
 *           and reads LBA 0 -- without ever touching the admin queue.
 *
 * This proves the primitive HOMI relies on: owning the controller in one
 * process and handing out I/O qpairs to others. Run as:
 *
 *   test_qpair_handover owner  <PCI-BDF> <descfile>   (process A, stays up)
 *   test_qpair_handover client <descfile>             (process B, reads LBA0)
 */

#define _UPCIE_WITH_NVME
#include <upcie/upcie.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct handover_file {
	char bdf[32];
	char region_path[256];
	uint64_t region_size;
	struct nvme_qpair_export qp;
};

static int
run_owner(const char *bdf, const char *descpath)
{
	struct hostmem_config config = {0};
	struct hostmem_heap heap = {0};
	struct nvme_controller ctrlr = {0};
	struct nvme_qpair ioq = {0};
	struct nvme_completion cpl = {0};
	struct nvme_command cmd = {0};
	struct handover_file hf = {0};
	char ready[320], stop[320];
	FILE *f;
	int err;

	err = hostmem_config_init(&config);
	if (err) {
		printf("FAILED: hostmem_config_init(); err(%d)\n", err);
		return err;
	}

	/* Heap backed by a shared (memfd) hugepage so the client can map the
	 * SQ/CQ rings the controller allocates from it. */
	err = hostmem_heap_init(&heap, 1024 * 1024 * 64ULL, &config);
	if (err) {
		printf("FAILED: hostmem_heap_init(); err(%d)\n", err);
		return err;
	}

	err = nvme_controller_open(&ctrlr, bdf, &heap);
	if (err) {
		printf("FAILED: nvme_controller_open(%s); err(%d)\n", bdf, err);
		goto out_heap;
	}

	/* Identify controller -- mirrors the existing single-process test and
	 * confirms the admin queue is live before we hand out an I/O qpair. */
	cmd.opc = 0x6;
	cmd.cdw10 = 1;
	err = nvme_qpair_submit_sync_contig_prps(&ctrlr.aq, ctrlr.heap, ctrlr.buf, 4096, &cmd,
						 ctrlr.timeout_ms, &cpl);
	if (err) {
		printf("FAILED: identify; err(%d)\n", err);
		goto out_ctrlr;
	}

	err = nvme_controller_create_io_qpair(&ctrlr, &ioq, 32);
	if (err) {
		printf("FAILED: nvme_controller_create_io_qpair(); err(%d)\n", err);
		goto out_ctrlr;
	}

	snprintf(hf.bdf, sizeof(hf.bdf), "%s", bdf);
	snprintf(hf.region_path, sizeof(hf.region_path), "%s", heap.memory.path);
	hf.region_size = heap.memory.size;
	nvme_qpair_export_from(&ioq, &heap, &hf.qp);

	f = fopen(descpath, "wb");
	if (!f || fwrite(&hf, sizeof(hf), 1, f) != 1) {
		printf("FAILED: writing descriptor '%s'\n", descpath);
		if (f)
			fclose(f);
		err = EIO;
		goto out_ctrlr;
	}
	fclose(f);

	snprintf(ready, sizeof(ready), "%s.ready", descpath);
	f = fopen(ready, "w");
	if (f) {
		fputs("ready\n", f);
		fclose(f);
	}

	printf("owner: ready qid=%u depth=%u sq_off=%" PRIu64 " cq_off=%" PRIu64 " region='%s'\n",
	       hf.qp.qid, hf.qp.depth, hf.qp.sq_offset, hf.qp.cq_offset, hf.region_path);
	fflush(stdout);

	/* Keep the controller and the shared region alive until the client is
	 * done (it drops a .stop sentinel) or a safety timeout elapses. */
	snprintf(stop, sizeof(stop), "%s.stop", descpath);
	for (int i = 0; i < 120; i++) {
		if (access(stop, F_OK) == 0)
			break;
		sleep(1);
	}

	err = 0;
out_ctrlr:
	nvme_controller_close(&ctrlr);
out_heap:
	hostmem_heap_term(&heap);
	return err;
}

static int
run_client(const char *descpath)
{
	struct hostmem_config config = {0};
	struct hostmem_heap heap = {0};       ///< client-local heap for data + PRPs
	struct hostmem_hugepage region = {0}; ///< imported shared region (rings)
	struct pci_func func = {0};
	struct nvme_qpair qp = {0};
	struct handover_file hf = {0};
	struct nvme_completion cpl = {0};
	struct nvme_command cmd = {0};
	const size_t bufsz = 4096;
	uint8_t *bar0;
	void *buf = NULL;
	char stop[320];
	uint8_t sc, sct;
	FILE *f;
	int err;

	f = fopen(descpath, "rb");
	if (!f || fread(&hf, sizeof(hf), 1, f) != 1) {
		printf("FAILED: reading descriptor '%s'\n", descpath);
		if (f)
			fclose(f);
		return EIO;
	}
	fclose(f);

	err = hostmem_config_init(&config);
	if (err) {
		printf("FAILED: hostmem_config_init(); err(%d)\n", err);
		return err;
	}

	err = hostmem_hugepage_import(hf.region_path, &region, &config);
	if (err) {
		printf("FAILED: hostmem_hugepage_import('%s'); err(%d)\n", hf.region_path, err);
		return err;
	}

	err = hostmem_heap_init(&heap, 1024 * 1024 * 16ULL, &config);
	if (err) {
		printf("FAILED: hostmem_heap_init(); err(%d)\n", err);
		goto out_region;
	}

	err = pci_func_open(hf.bdf, &func);
	if (err) {
		printf("FAILED: pci_func_open('%s'); err(%d)\n", hf.bdf, err);
		goto out_heap;
	}
	err = pci_bar_map(func.bdf, 0, &func.bars[0]);
	if (err) {
		printf("FAILED: pci_bar_map(BAR0); err(%d)\n", err);
		goto out_func;
	}
	bar0 = func.bars[0].region;

	err = nvme_qpair_import(&qp, &hf.qp, region.virt, bar0, &heap);
	if (err) {
		printf("FAILED: nvme_qpair_import(); err(%d)\n", err);
		goto out_func;
	}

	buf = hostmem_dma_malloc(&heap, bufsz);
	if (!buf) {
		err = errno;
		printf("FAILED: hostmem_dma_malloc(); err(%d)\n", err);
		goto out_qp;
	}
	memset(buf, 0, bufsz);

	/* Read LBA 0, a single logical block, into the client-local buffer. */
	cmd.nsid = 1;
	cmd.opc = 0x2; ///< READ
	cmd.cdw10 = 0; ///< SLBA low == 0
	cmd.cdw12 = 0; ///< NLB == 0 (zero-based: one block)

	err = nvme_qpair_submit_sync_contig_prps(&qp, &heap, buf, bufsz, &cmd, 5000, &cpl);
	if (err) {
		printf("FAILED: nvme_qpair_submit_sync_contig_prps(); err(%d)\n", err);
		goto out_buf;
	}

	sc = (cpl.status & 0x1FE) >> 1;
	sct = (cpl.status & 0xE00) >> 8;
	if (sc) {
		printf("FAILED: read status sct(0x%x) sc(0x%x)\n", sct, sc);
		err = EIO;
		goto out_buf;
	}

	printf("client: read LBA0 on handed-out qpair qid=%u; first 16 bytes:", hf.qp.qid);
	for (int i = 0; i < 16; i++)
		printf(" %02x", ((unsigned char *)buf)[i]);
	printf("\nSUCCESS: client drove a handed-out qpair with no admin-queue access\n");

out_buf:
	hostmem_dma_free(&heap, buf);
out_qp:
	nvme_qpair_import_term(&qp);
out_func:
	pci_func_close(&func);
out_heap:
	hostmem_heap_term(&heap);
out_region:
	hostmem_hugepage_free(&region);

	/* Tell the owner it can shut down. */
	snprintf(stop, sizeof(stop), "%s.stop", descpath);
	f = fopen(stop, "w");
	if (f) {
		fputs("stop\n", f);
		fclose(f);
	}
	return err;
}

int
main(int argc, char **argv)
{
	if (argc == 4 && strcmp(argv[1], "owner") == 0)
		return run_owner(argv[2], argv[3]);
	if (argc == 3 && strcmp(argv[1], "client") == 0)
		return run_client(argv[2]);

	fprintf(stderr,
		"usage:\n"
		"  %s owner  <PCI-BDF> <descfile>\n"
		"  %s client <descfile>\n",
		argv[0], argv[0]);
	return 2;
}
