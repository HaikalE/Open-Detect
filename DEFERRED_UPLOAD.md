# Training di Drive worker, upload A lewat cell CPU

Notebook hanya mengubah satu code cell (`grouped-9`, training) dan menambahkan
satu code cell terakhir (`deferred-upload`). Cell setup, dependensi, dataset,
model/loss/seed dan commit ilmiah c729403f7fd4d1207522e3ca793cafb5f8cb8eb0 tetap.
Pengantar lama masih menjelaskan relay per epoch; komentar pada cell training
baru menyatakan bahwa penyimpanan sekarang mengikuti alur di bawah.
Layanan Apps Script A yang sudah ada tidak perlu diubah/deploy ulang.

## Pakai

1. Selesaikan/hentikan sesi lama dahulu; gunakan notebook terbaru untuk sesi berikutnya.
2. GPU: Run all menjalankan setup lama dan training baru. Checkpoint aktif tersimpan
   di My Drive akun worker dalam `OpenDetect_WORKER_<scenario>`. Folder menggunakan
   `CURRENT.json` untuk memetakan path checkpoint ke ID file yang terverifikasi.
3. Run all melewati upload jika GPU masih terdeteksi, sehingga upload tidak
   menghabiskan sesi GPU setelah training selesai/pause.
4. Hentikan training; ganti runtime CPU; jalankan cell terakhir saja.
   Autentikasi akun worker yang sama dan izinkan Secrets relay yang sama. Tidak
   perlu instal environment training, memuat PyTorch, atau download dataset.
   `PREVIEW_ONLY=False` langsung upload. Set True jika hanya ingin melihat daftar.
5. `CLEAN_COMPLETED_AFTER_UPLOAD=True` menghapus permanen **resume state fold selesai**
   setelah best-model, hasil, scores, config dan completion marker A diverifikasi.
   Best-model, hasil, log, serta dua slot resume fold aktif tetap di worker.

## Retensi dan kegagalan

Trainer tetap menulis dua slot bergantian: `last.pt` dan `last.backup.pt`.
Model terbaik disimpan terpisah, satu per fold. Urutan: checkpoint lokal selesai,
upload worker, verifikasi SHA256/ukuran/pemilik melalui API, publish manifest,
readback manifest, hapus ID versi tergantikan, baru izinkan epoch berikutnya.
Cleanup menyimpan daftar ID pending supaya retry setelah koneksi putus idempotent.
File asing/orphan tidak dipindai untuk penghapusan. Failure saat upload/publish
mempertahankan snapshot cloud sebelumnya. Failure cleanup menghentikan training
dengan snapshot terbaru sudah tersimpan. File staging yang tak terkomit dari
upload gagal mungkin tetap memakai kuota dan memerlukan pemeriksaan terpisah.

Ukuran A-2 yang sebelumnya diamati: dua slot + best sekitar 975 MB per fold aktif,
ditambah headroom upload sementara dan artefak fold selesai. Preflight menyisakan
1,5 GB di luar payload incoming dan launcher memeriksa ruang disk lokal. Simpan
ke worker masih memakai bandwidth; percepatan nyata perlu diukur di Colab.

Upload A mengirim current state saja. Checkpoint fold selesai tidak diunduh/dikirim
ulang jika bukti completion lengkap. Upload ulang state sama tidak membuat
generation atau objek A baru. A tetap memakai service immutable lama: upload
berbeda pada sesi berbeda menambah versi A, sehingga arsip lama A perlu GC terpisah.
Notebook ini tidak membersihkan arsip A yang sudah ada.

Hanya satu runtime aktif per skenario; lease service lama dipakai selama training
dan transfer. Setelah pause, snapshot terbaru berada di worker. **Upload dahulu
sebelum berpindah ke akun/worker lain**: layanan lama tidak menyimpan reservasi
antar-sesi. Jika runtime mati dengan lease tertinggal, A memastikan proses lama
berhenti lalu memakai prosedur releaseLostWorker yang sudah ada. Riwayat A/worker
berbeda ditolak; tidak dipilih otomatis berdasarkan tanggal atau nama file.

`SESSION_HOURS=3` adalah anggaran launcher, bukan jaminan durasi Colab. Pergantian
runtime bisa menghilangkan file lokal; resume menggunakan checkpoint cloud worker.

## Build dan verifikasi

`python build_worker_notebooks.py <folder-output>` menghasilkan delapan notebook.
`update_notebook()` dapat memperbarui notebook Drive asli sambil mempertahankan
cell lain/output yang tersimpan. Modul helper disematkan di kedua cell agar cell
upload tetap mandiri pada runtime CPU baru.

Tes lokal: failure upload/commit, retensi dua slot, cleanup berulang, checkpoint
aktif terlindungi, riwayat berbeda ditolak, transfer CPU diulang tanpa duplikasi,
dan subprocess tidak melanjutkan epoch sebelum ACK cloud. Tes memakai Drive tiruan.
Resume GPU dan transfer Google nyata tetap perlu diamati pada penggunaan berikutnya.
