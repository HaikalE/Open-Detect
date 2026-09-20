# Pilot temporal fusion — mulai di sini

Status: **P0 / audit data CPU**, bukan training fusion dan bukan pengganti hasil replikasi.

Jalankan `01_AUDIT_DATA_CPU.ipynb` di Colab, runtime CPU, dari atas ke bawah.
Gunakan akun yang diberi akses ke folder sumber di Drive A. Akses berdasarkan ID
folder, bukan pencarian file pada akun yang kebetulan login. Tidak perlu relay secret.
Notebook mandiri; tidak membutuhkan clone GitHub atau GPU. Kode audit dan
preprocessing disertakan dengan SHA-256 agar snapshot notebook dapat ditelusuri.

## Yang dilakukan notebook

1. Autentikasi dan cek izin folder sumber serta folder pilot.
2. Audit keenam NPZ (A/USTC, B/Malicious_TLS, C/combined); batas unduh total 400 MiB.
3. Inventaris PCAP terbatas (maksimum 3.000 entri/100 folder) dan unduh maksimal
   enam PCAP <= 8 MiB, secara round-robin antar kelompok sumber yang ditemukan.
   Inventaris parsial dilabeli; tidak disamakan dengan file tidak tersedia.
4. Cek timestamp, satu/multi-biflow, preprocessing 8 paket, kandidat kecocokan
   hash citra. Raw capture multi-flow harus disessionisasi sebelum dipasangkan.
5. Simpan laporan JSON kecil ke folder pilot lewat cell terakhir terpisah.
   Cell ini tidak menghapus atau menimpa checkpoint/dataset/replikasi.

Tidak ada klaim seluruh flow sudah berpasangan hanya karena beberapa hash cocok.
Tidak ada dugaan mapping berdasarkan urutan baris CSV. NPZ hanya `data,target`
tidak cukup untuk membuktikan mapping. IAT harus berasal dari timestamp PCAP.

## Urutan lanjut

- P0 sekarang: audit sumber dan pemetaan, baca `training_ready: false` sebagai
  **belum melewati pemeriksaan pairing**, bukan kegagalan replikasi.
- P1: bangun manifest paired lengkap; jika mapping NPZ lama tidak dapat dibuktikan,
  bangun ulang dataset berpasangan dan jalankan ulang baseline pada sampel/split sama.
- P2: implementasi fusion dan uji kecil satu skenario/fold, lalu ablation B0/E1/E2.
- P3: setelah pilot lolos baru seluruh skenario/fold. Tidak langsung 5-fold semua.

Plan implementasi dan rujukan: `../docs/TEMPORAL_FUSION_PLAN.md`.
