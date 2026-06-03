// SPDX-License-Identifier: BSD-3-Clause

/**
 * Controller-attach owner
 * =======================
 *
 * Owns an NVMe controller in userspace, pre-creates a pool of I/O qpairs whose
 * SQ/CQ rings live in a shared (memfd) hugepage, captures the Identify
 * Controller and Identify Namespace payloads, and writes a upcie_attach_desc
 * file. It then stays alive so the controller and shared region remain up while
 * a client (e.g. the xNVMe upcie backend in attach mode) drives the qpairs.
 *
 * This stands in for the HOMI daemon's controller-owning role until HOMI itself
 * provides it. Run as:
 *
 *   test_attach_owner <PCI-BDF> <descfile> <nqpairs>
 */

#define _UPCIE_WITH_NVME
#include <upcie/upcie.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int
identify(struct nvme_controller *ctrlr, uint32_t nsid, uint32_t cns, void *out)
{
	struct nvme_completion cpl = {0};
	struct nvme_command cmd = {0};

	cmd.opc = 0x6; ///< IDENTIFY
	cmd.nsid = nsid;
	cmd.cdw10 = cns;
	memset(out, 0, UPCIE_ATTACH_IDFY_NBYTES);

	return nvme_qpair_submit_sync_contig_prps(&ctrlr->aq, ctrlr->heap, out,
						  UPCIE_ATTACH_IDFY_NBYTES, &cmd, ctrlr->timeout_ms,
						  &cpl);
}

int
main(int argc, char **argv)
{
	struct hostmem_config config = {0};
	struct hostmem_heap heap = {0};
	struct nvme_controller ctrlr = {0};
	struct nvme_qpair *qpairs = NULL;
	struct upcie_attach_desc *desc = NULL;
	const uint32_t nsid = 1;
	char ready[320], stop[320];
	uint32_t nqpairs;
	const char *bdf, *descpath;
	int err;

	if (argc != 4) {
		fprintf(stderr, "usage: %s <PCI-BDF> <descfile> <nqpairs>\n", argv[0]);
		return 2;
	}
	bdf = argv[1];
	descpath = argv[2];
	nqpairs = (uint32_t)atoi(argv[3]);
	if (nqpairs < 1 || nqpairs > UPCIE_ATTACH_MAX_QPAIRS) {
		fprintf(stderr, "nqpairs must be in [1, %d]\n", UPCIE_ATTACH_MAX_QPAIRS);
		return 2;
	}

	desc = calloc(1, sizeof(*desc));
	qpairs = calloc(nqpairs, sizeof(*qpairs));
	if (!desc || !qpairs) {
		fprintf(stderr, "FAILED: calloc\n");
		return ENOMEM;
	}

	err = hostmem_config_init(&config);
	if (err) {
		printf("FAILED: hostmem_config_init(); err(%d)\n", err);
		return err;
	}

	/* Shared (memfd) hugepage so the client can map the SQ/CQ rings. */
	err = hostmem_heap_init(&heap, 1024 * 1024 * 128ULL, &config);
	if (err) {
		printf("FAILED: hostmem_heap_init(); err(%d)\n", err);
		return err;
	}

	err = nvme_controller_open(&ctrlr, bdf, &heap);
	if (err) {
		printf("FAILED: nvme_controller_open(%s); err(%d)\n", bdf, err);
		goto out_heap;
	}

	/* Identify must DMA into heap-backed memory; ctrlr.buf is a 4 KiB DMA
	 * buffer the controller allocated. Copy the result into the descriptor. */
	err = identify(&ctrlr, 0, 1, ctrlr.buf); ///< CNS 1: Identify Controller
	if (err) {
		printf("FAILED: identify controller; err(%d)\n", err);
		goto out_ctrlr;
	}
	memcpy(desc->idfy_ctrlr, ctrlr.buf, UPCIE_ATTACH_IDFY_NBYTES);

	err = identify(&ctrlr, nsid, 0, ctrlr.buf); ///< CNS 0: Identify Namespace
	if (err) {
		printf("FAILED: identify namespace; err(%d)\n", err);
		goto out_ctrlr;
	}
	memcpy(desc->idfy_ns, ctrlr.buf, UPCIE_ATTACH_IDFY_NBYTES);

	for (uint32_t i = 0; i < nqpairs; i++) {
		err = nvme_controller_create_io_qpair(&ctrlr, &qpairs[i], 1024);
		if (err) {
			printf("FAILED: create_io_qpair(%u); err(%d)\n", i, err);
			goto out_ctrlr;
		}
		nvme_qpair_export_from(&qpairs[i], &heap, &desc->qpairs[i]);
	}

	snprintf(desc->bdf, sizeof(desc->bdf), "%s", bdf);
	snprintf(desc->region_path, sizeof(desc->region_path), "%s", heap.memory.path);
	desc->region_size = heap.memory.size;
	desc->nsid = nsid;
	desc->nqpairs = nqpairs;

	err = upcie_attach_write(descpath, desc);
	if (err) {
		printf("FAILED: upcie_attach_write('%s'); err(%d)\n", descpath, err);
		goto out_ctrlr;
	}

	snprintf(ready, sizeof(ready), "%s.ready", descpath);
	FILE *rf = fopen(ready, "w");
	if (rf) {
		fputs("ready\n", rf);
		fclose(rf);
	}

	printf("owner: ready nqpairs=%u nsid=%u region='%s'\n", nqpairs, nsid, desc->region_path);
	fflush(stdout);

	snprintf(stop, sizeof(stop), "%s.stop", descpath);
	for (int i = 0; i < 180; i++) {
		if (access(stop, F_OK) == 0)
			break;
		sleep(1);
	}

	err = 0;
out_ctrlr:
	nvme_controller_close(&ctrlr);
out_heap:
	hostmem_heap_term(&heap);
	free(desc);
	free(qpairs);
	return err;
}
