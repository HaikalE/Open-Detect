# Mini open-set training pilot

This short run follows the successful CPU smoke notebook. It trains B0
(image-only), E1 (payload length + direction), and E2 (E1 + IAT) on the same
flow-disjoint cohort and seed. It uses the original OpenDetect loss and training
defaults for learning rate/lambda, with an eight-epoch cap and validation-based
early stopping. The B0 image trunk, decoder, and prototypes initialize all arms
identically. Best weights remain in RAM and are discarded after metrics; only a
small JSON result is uploaded to Drive.

Threshold is calibrated to 95% acceptance using known validation scores only.
Unknown Tinba is held out for test and cannot affect selection. The report
includes known closed-set scores plus open-set AUROC, known acceptance, unknown
rejection, training curves, and runtime for one seed.

Run `04_MINI_OPENSET_TRAIN_CPU_V2.ipynb` from a fresh CPU or GPU Colab runtime. It
requires the verified P2 cohort files and checks their checksums before training.
Scope remains four known classes plus one unknown from bounded PCAP prefixes.
Treat all metrics as feasibility results, not as scenario A-1 or thesis claims.
