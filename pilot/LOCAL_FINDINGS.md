# Local preflight — 2026-09-20

These observations concern local `OpenDetect_Colab/data`, not an executed Colab
audit or a claim that every Drive copy has identical content.

- All six NPZ files load without pickle and have only `data` and `target`.
  Images are uint8 32x32. No explicit flow/PCAP IDs are carried in these files.
- USTC: 31,126 train + 3,459 test rows; Malicious_TLS: 80,945 + 34,691;
  combined: 112,071 + 38,150. Do not sum these as independent samples.
- Two inspected USTC captures, Facetime and Tinba, contain multiple biflow tuples.
  They cannot be treated as a single temporal sequence. Sessionization and mapping
  back to image samples are needed. This is a data-preparation gate, not a failed
  replication result.
- Ten new unit/notebook-syntax tests pass, including reversed direction, original
  image-preprocessor compatibility, multiflow refusal, invalid timestamp, scan
  limit, duplicate row preservation and corrupt NPZ handling.
- Notebook authentication/download/upload has not been executed in a live Colab
  session. The user runs that part using an account with source-folder access.

Current delivery intentionally includes only the runnable P0 CPU audit notebook.
It does not claim a implemented/trained BiGRU model. P1 pairing precedes P2 training.
