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
