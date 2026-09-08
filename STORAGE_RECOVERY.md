# OpenDetect: penyimpanan A dan pemulihan B

Perubahan ini hanya launcher/penyimpanan, bukan model, dataset, indeks, loss,
epoch, seed, atau evaluasi. Training tetap commit
`c729403f7fd4d1207522e3ca793cafb5f8cb8eb0`.

## Notebook training yang sama

Delapan notebook GROUPED diperbarui di ID Drive yang sama setelah salinan
cadangan dibuat. Salinan yang sudah diunduh ke akun B tidak ikut berubah.
Jangan memuat ulang atau remount notebook saat producer training aktif.

Sebelum training, launcher memeriksa akun API terhadap pemilik folder origin,
kemudian menguji penulis mount melalui file probe kecil dan metadata Drive.
Jika akun salah, path hanya folder bernama sama, atau cloud tidak mengonfirmasi,
launcher berhenti sebelum training. Cadangan kuota awal minimum 6 GB diperiksa.
Ini tidak menjamin sinkronisasi per epoch dan tidak mengubah protokol checkpoint.
Jika Colab tidak mengizinkan autentikasi A dari sesi B, gunakan sesi A; launcher
ini tidak menyediakan uploader lintas akun otomatis.

## Satu notebook pemulihan, tiga tahap

Gunakan `OpenDetect_RECOVERY_B_TO_A.ipynb`, CPU cukup. Default hanya inventaris.

Versi `root-metadata-v2` mengunduh `OpenDetect_B_root_candidates.json` otomatis:
checkpoint, JSON pendamping, progress, log, hasil bernama run dan ringkasan.
Nama saja bukan bukti keterkaitan eksperimen; ID tetap harus dipilih.
File `.writing` ditandai parsial, bukan otomatis valid untuk resume.
Laporan kandidat bukan manifest backup; manifest kosong kini ditolak.

1. Login B, `PHASE='INVENTORY_B'`, jalankan. Tinjau daftar scoped outputs.
   Kandidat `last.pt` di root belum dipilih: masukkan hanya ID file eksperimen
   yang benar ke `EXTRA_ORPHAN_IDS`. Jangan memilih seluruh root berdasarkan nama.
2. Setelah producer berhenti (bukan menghapus runtime/cache yang belum tersinkron),
   isi `STOPPED='SEMUA RUN SUMBER SUDAH BERHENTI'`,
   `SHARE_AND_EXPORT=True`, jalankan konfigurasi dan tahap lagi. Unduh manifest.
3. Otorisasi A pada runtime pemulihan baru, notebook yang sama:
   `PHASE='BACKUP_A'`, isi STOPPED, jalankan dan upload manifest tadi.
   Backup dibuat server-side di THESIS IMPLEMENTASI/RECOVERY_B_<identitas>.
   Pemilik A, ukuran, checksum MD5 provider, dan sumber stabil harus cocok.
   Simpan receipt JSON yang diunduh. Copy tidak memindahkan file sumber.
4. Otorisasi B, `PHASE='CLEANUP_B'`, upload receipt, PREVIEW dahulu.
   Pilih ID sumber dalam `SELECTED_IDS`. Untuk trash, isi `CLEANUP_ACTION='TRASH'`
   dan `CONFIRMATION='PINDAHKAN FILE TERVERIFIKASI KE SAMPAH'`, serta STOPPED.
5. Sampah MASIH memakai kuota. Hanya setelah backup diperiksa, jalankan terpisah
   `CLEANUP_ACTION='DELETE_PERMANENT'` dan
   `CONFIRMATION='HAPUS PERMANEN FILE TERVERIFIKASI'` untuk ID tadi.
   File harus sudah di Sampah dan backup A harus tetap valid. Penghapusan ini
   tidak dapat dipulihkan dari B. Tidak ada operasi kosongkan seluruh Sampah.

Autentikasi akun A/B diperlukan oleh pengguna; pembuatan notebook ini belum
berarti backup atau pembersihan B sudah dieksekusi.

## Batas penting

- Alat hanya melihat file CLOUD. Pesan CHECKPOINT COMMITTED dari FUSE bukan bukti
  cloud selesai upload saat quota exceeded. Jangan terminate runtime sebelum
  salinan checkpoint yang bisa dipulihkan benar-benar tersedia.
- Backup terpisah tidak menimpa outputs canonical A. Manifest menyimpan path asal,
  nama backup ditambahi ID untuk mempertahankan orphan yang namanya sama.
  Mengaktifkan resume dari backup perlu pencocokan slot checkpoint dan metadata;
  notebook ini tidak otomatis menganggap semua file bernama last.pt valid.
- File sumber berubah, backup hilang, pemilik salah, atau checksum beda: cleanup
  berhenti. Sumber harus tidak ditulis selama backup/pembersihan; Drive tidak
  menyediakan transaksi atomik lintas seluruh daftar file.
- Uji lokal memakai API tiruan, bukan menghapus data Drive sungguhan.

Referensi: [kuota Drive](https://support.google.com/drive/answer/2424368?hl=en),
[kepemilikan](https://support.google.com/drive/answer/2494892?hl=en),
[trash dan delete API](https://developers.google.com/workspace/drive/api/guides/delete).
# Update 2026-09-08 — cloud-workflow-v4 (menggantikan langkah manual di bawah)

Notebook recovery yang sama sekarang dibangun dengan `build_cloud_recovery.py`.
Tidak ada `files.upload`/`files.download`. Manifest dan receipt dibaca langsung
dari Drive API. CPU cukup; tidak menjalankan training atau mengatasi limit GPU.

1. **B / PREPARE_B**: preview outputs dan kandidat root, pilih ID sumber persis
   di SOURCE_IDS, konfirmasi semua producer berhenti. Bagikan sumber ke A dan
   simpan manifest/inspection ke folder handoff di THESIS IMPLEMENTASI.
2. **A / BACKUP_A**: baca satu handoff (pilih JOB_FOLDER_ID bila ambigu), copy
   server-side ke folder arsip A, periksa owner/ukuran/checksum, simpan receipt.
3. **A / PLAN_A**: baca RECEIPT_ID. Skenario/fold/seed dikenali dari metadata
   format 2, bukan nama file. SHA256 memasangkan PT dan metadata; konfigurasi
   canonical (termasuk kode/data/split) harus cocok dengan experiment_signature.
   Cetak tujuan dan link notebook training yang tepat.
4. **A / INSTALL_A**: pilih satu skenario/fold hasil plan. Semua producer A/B
   wajib berhenti. KEEP_A bila A sama/lebih baru; epoch sama dengan hash berbeda
   ditolak. Copy bertahap, verifikasi, pertahankan file lama dengan nama BEFORE_,
   publish PT kemudian JSON. Slot valid paling baru yang lain tidak disentuh.
5. **B / PREVIEW_B → TRASH_B**: periksa receipt/backup, tinjau daftar, salin token
   daftar dan ketik konfirmasi. Hanya sumber receipt, tidak menghapus backup A.
6. **B / DELETE_B**: tahap terpisah setelah resume training nyata berhasil di A,
   disertai pengakuan pengguna dan konfirmasi permanen. Backup diverifikasi lagi.
   Sumber wajib sudah di Sampah. File 404 dilaporkan sebagai tidak tersedia,
   bukan diklaim sudah dihapus; file lain tetap diverifikasi satu per satu.

Untuk kasus C-1 yang sudah diarsipkan: RECEIPT_ID lama sudah terisi. Mulai PLAN_A;
tidak perlu menyiapkan B lagi. Untuk pekerjaan baru kosongkan ID lama. Pemilihan
otomatis hanya ketika hasil discovery tepat satu; jika lebih dari satu, salin ID
yang tepat, tetap tanpa upload JSON. Konfirmasi akun tidak diotomatisasi.

**Batas:** config/folder canonical yang belum ada tidak dibuat dari tebakan.
Completed marker tidak ditimpa. Log/hasil/data tetap di arsip, bukan otomatis
dianggap output canonical. Metadata tidak berpasangan dilaporkan. Alat tidak
memuat pickle/model/RNG; trainer harus membuktikan resume nyata. Sampah masih
memakai kuota. Tidak ada transaksi lintas file: gunakan satu sesi recovery dan
jangan menulis bersamaan. STAGED_/BEFORE_ dipertahankan, tidak dibersihkan otomatis.
Identitas API menentukan pemilik file baru; berbagi folder sebagai Editor bukan
pemindahan kepemilikan. Pengaman training A-only yang ada tidak diubah.

Uji menggunakan API tiruan mencakup pemasangan normal, putus saat publish,
pencegahan rollback, riwayat berbeda, identitas salah, dan pengaman cleanup.
Penerbitan notebook bukan bukti operasi A/B atau resume GPU sudah dijalankan.

---

## Dokumentasi versi lama (arsip; bukan langkah notebook v4)
