# Gerbang mini fusion USTC (uji kelayakan, bukan evaluasi tesis)

Hasil lokal pada branch `codex/temporal-fusion`:

- 564 citra multi-paket dengan IAT positif dari tujuh prefix PCAP (masing-masing
  maksimum 10.000 paket fisik). Dataset berpasangan menyimpan citra, urutan
  `transport_payload_bytes`, arah, IAT detik, panjang, mask, label, dan split.
- Mini holdout memakai empat kelas known (Geodo, Miuref, WorldOfWarcraft,
  Zeus) dan Tinba sebagai unknown **hanya di test**. Ini bukan skenario A-1:
  kelas known lain belum tercakup.
- Split baru dilakukan per kelas dan flow group dengan seed 2026. Hitungan:

| Kelas | Train | Val | Test |
| --- | ---: | ---: | ---: |
| Geodo | 71 | 15 | 15 |
| Miuref | 133 | 27 | 27 |
| WorldOfWarcraft | 106 | 23 | 22 |
| Zeus | 35 | 8 | 8 |
| Tinba (unknown) | 0 | 0 | 74 |

Tiga flow yang sebelumnya melintasi partisi NPZ lama kini tidak dipecah oleh
split baru. Builder menolak hash citra/label salah, IAT yang tidak sesuai
timestamp, duplikat citra, pemakaian ulang paket sumber, dan panjang/mask yang
tidak cocok. Semua artefak tetap diberi label **NOT_THESIS_EVAL**.

Model eksperimental `model_fusion.py` mempertahankan decoder, prototype Gaussian,
dan komponen loss OpenDetect. `networks/temporal.py` memakai BiGRU bidirectional
yang mengabaikan padding melalui panjang sequence. B0 memakai model image-only
asli pada cohort sama; E1 membaca payload length + direction; E2 menambahkan
IAT. Ini implementasi proposal, **bukan** klaim arsitektur persis MalDIST atau
DISTILLER. Model lama tidak diganti.

Tes integrasi satu batch CPU berhasil untuk B0/E1/E2; output one-batch loss
**tidak boleh dibandingkan sebagai skor performa**. Normalisasi `log1p` untuk
payload length dan IAT di-fit hanya dari paket train known; padding tetap nol.
Belum ada model terlatih, threshold unknown, accuracy, F1, atau AUROC.

Reproduksi lokal:

1. `python -m pilot.build_mini_cohort <report-kelas.json> ... --dataset-dir
   <folder-NPZ> --output-dir <folder-baru>`
2. `python -m pilot.smoke_mini_fusion --cohort-dir <folder-baru> --output
   <file-json-baru>`

Sebelum eksperimen tesis: perlu pemulihan dan audit seluruh kelas/capture yang
diperlukan, split/fold final yang dikunci, serta B0/E1/E2 dan kontrol kapasitas
image-only pada sampel, seed, budget, dan horizon paket yang sama. Threshold
unknown dikalibrasi memakai validasi known saja, tanpa menyentuh Tinba test.
