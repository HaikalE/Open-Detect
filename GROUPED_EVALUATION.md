# Evaluasi grouped tanpa ablasi

Notebook terpisah ini membaca eksperimen `OpenDetect_GROUPED_2026-09-06`.
Training tetap commit `c729403f7fd4d1207522e3ca793cafb5f8cb8eb0` pada
`codex/grouped-split-colab`; reporter berada pada `codex/grouped-evaluation`.
Tidak ada perubahan model, optimizer, training, dataset, threshold, atau notebook training.

## Jalankan

1. Tunggu minimal satu fold selesai training 100 epoch **dan** test otomatisnya selesai.
2. Buka `OpenDetect_GROUPED_EVALUATION_NO_ABLATION.ipynb` di Colab, lalu Run all.
3. CPU dapat dipakai. GPU hanya opsional untuk diagnostik; jangan hentikan training lain.
4. Izinkan mount Drive. Lokasi input default: `MyDrive/THESIS IMPLEMENTASI/OpenDetect_GROUPED_2026-09-06/outputs`.
5. Baca `status.csv` dan `diagnostics_status.csv` sebelum mengutip laporan.
6. Hasil/ZIP ada di folder `evaluation_reports`, terpisah dari `outputs`.
   Ekstrak ZIP lalu buka `REPORT.html`. Run all lagi menghasilkan laporan baru, bukan menimpa laporan lama.

## Isi

| Hasil | Sumber dan batas |
| --- | --- |
| Per-run accuracy, precision, recall, F1, AUROC | Scores NPZ diverifikasi ulang terhadap JSON, bukan training accuracy |
| Mean ± sample SD | Lima seed 2022–2026 lengkap; tidak memilih atau membuang seed jelek |
| Referensi angka P01 Tabel V–VII | Grouped split berbeda; averaging F1 paper belum terkonfirmasi, jadi bukan perbandingan identik |
| Known-class weighted F1 | Kelas known per skenario; bukan menggantikan eksperimen closed-world semua kelas Tabel IV |
| Distribusi skor / threshold | Fig.4-style, fold 0; threshold berasal dari known-validation 95%, tidak dituning memakai unknown test |
| ROC dan confusion counts | Diagnostik tambahan, bukan klaim reproduksi figur tertentu |
| t-SNE encoder mean | Fig.5-style, fixed seed 2022 dan subsampel seimbang; algoritme proyeksi merupakan pilihan implementasi |
| Absolute pixel gradient | Fig.6-style, contoh known ditetapkan secara deterministik, tidak menyimpulkan field TLS |
| Waktu inferensi | Full forward termasuk decoder, batch 1, input sudah di device, warmup 10; bukan FET PCAP atau benchmark identik V100 |

Seluruh delapan skenario dicatat; hanya run dengan completed marker, identitas yang
sesuai, dan hash checkpoint/result/scores yang valid masuk tabel. Tidak perlu
menunggu semua skenario selesai untuk melihat hasil per-run. Rata-rata parsial
tidak disajikan sebagai hasil lima seed.

Model diagnostics hanya memakai fold 0/seed 2022 (maksimum 40 contoh per kelas
asal) untuk membatasi biaya dan menghindari pemilihan seed terbaik. NPZ latent
menyimpan nomor baris relatif partition, label asli, label reindex, dan proyeksi.
File metadata menyimpan hash checkpoint dan identitas split yang menunjuk indeks
baris asli. Threshold tetap; diagram tidak digunakan untuk mengoptimalkan model.

Tidak ada ablation, pelatihan baseline lain, PCAP extraction benchmark, atau
eksperimen baru closed-world semua kelas. Tidak mengklaim seluruh evaluasi paper
telah direplikasi. Angka waktu/akurasi hanya dihasilkan ketika artefak riil tersedia.

## Keamanan run

Reporter hanya membaca folder training. Folder report harus berada di luar input
dan checkout source. File report tidak menimpa report sebelumnya. Training aktif
tanpa completed marker menjadi pending. Integritas gagal menjadi invalid dan
ditampilkan sebagai error, bukan diam-diam dilewati sebagai hasil sah.
Notebook tidak menjadwalkan training, tidak mengubah checkpoint, dan tidak
memerlukan pemindahan runtime training ke notebook evaluasi.

Referensi: P01, DOI [10.1109/TIFS.2025.3612141](https://doi.org/10.1109/TIFS.2025.3612141),
PDF pp.11–14. Tabel V–VII diambil sebagai nilai referensi utama; tabel ablasi tidak digunakan.
