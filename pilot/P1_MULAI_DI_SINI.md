# P1 — sessionization & pairing pilot

Jalankan **02_PAIRING_PILOT_CPU.ipynb**, CPU, semua cell berurutan.
Tidak perlu mengulang P0. Login dengan akun yang punya akses ke sumber dan folder
pilot Drive A. Cell terakhir mengunggah hasil ke subfolder baru `P1_<run-id>`;
pengulangan cell upload memverifikasi checksum, tidak menimpa file berbeda.

## Cakupan

- Keenam NPZ diperiksa untuk pencocokan; label USTC, Malicious_TLS dan combined
  tidak dicampur sebagai satu namespace. Offset combined USTC +24 berasal dari
  `data/splits.py`, bukan hasil tebakan dari urutan baris.
- Empat PCAP USTC kecil: Facetime, Tinba, Skype, BitTorrent. Ini sampel engineering,
  bukan seluruh kelas/fold dan bukan dataset untuk evaluasi tesis.
- Arsip Malicious_TLS dibaca daftar isinya saja. Pemeriksaan lokal menemukan hanya
  `malicious_TLS.csv`. Notebook memverifikasi arsip Drive secara terpisah.
  CSV agregat tidak menggantikan raw PCAP dengan timestamp per paket.

## Hasil

`pairing_summary.json`: jumlah kandidat, kecocokan, error, scan parsial dan arsip.
Bagian `temporal_coverage` menunjukkan panjang sequence serta jumlah kandidat cocok
yang benar-benar memiliki lebih dari satu paket dan IAT positif. Satu paket tidak
mempunyai informasi jeda antarpaket; jangan lanjut training temporal dari itu.
`candidate_manifest.jsonl`: source SHA, ID sesi, indeks/timestamp paket, hash citra,
label yang diharapkan, seluruh baris NPZ yang cocok, tanpa pembagian split.
`paired_candidates_NOT_TRAIN_READY.npz`: citra, sequence, panjang, mask; indeks
baris cocok dengan `array_row` manifest. Tidak mengandung target label terverifikasi.
IAT dihitung sebelum padding, dari selisih timestamp desimal, tanpa normalisasi.
IP/port tidak dimasukkan sebagai fitur sequence atau ditulis ke manifest.

Aturan kandidat: bidirectional 5-tuple (plus versi IP), idle >60s, SYN baru dengan
sequence berbeda (retransmisi SYN sama tidak memisah), dan sesudah RST. FIN tidak
langsung menutup sesi agar ACK tidak terpisah. Ini heuristik eksplisit yang masih
harus dibandingkan dengan proses pembentukan data asli. Window hanya first-8;
sliding-window augmentation belum dicari. Maksimum 200.000 paket/10.000 sesi per
capture; batas tercapai dilaporkan sebagai parsial, bukan cakupan lengkap.

Makna status:

- `UNIQUE_IN_PILOT_ONLY`: satu kandidat dan satu baris USTC cocok dalam cakupan
  pilot; **belum** terbukti unik di seluruh sumber atau aman untuk split.
- `AMBIGUOUS_HASH`: lebih dari satu kandidat/baris atau sumber USTC tidak unik;
  tidak memilih salah satu secara acak.
- `LABEL_CONFLICT`: namespace/label yang cocok berbeda dari label capture kandidat.
- `NO_MATCH`: first-8 dengan kebijakan saat ini tidak cocok. Bukan bukti bahwa
  semua citra mustahil dipetakan; cek tool/sessionization/preprocessing/window asli.

## Gerbang lanjut

`training_ready=false` tetap disengaja: hasil belum boleh otomatis dipakai melatih
BiGRU. Verifikasi sumber label, collision seluruh capture, aturan sesi/window asli,
dan split antarsesi/duplikat dulu. Jika perlu membangun ulang paired dataset,
baseline citra harus dilatih ulang pada sampel dan split sama untuk pembandingan
fusion yang adil. Replikasi lama tetap sah sebagai eksperimen terpisah.

Untuk B/C, cari PCAP Malicious_TLS asli bila arsip hanya CSV. Jangan mengarang IAT
atau memakai urutan CSV sebagai urutan waktu. Tidak perlu mengumpulkan paper baru
untuk mengatasi kekurangan data sumber ini.

## Uji lokal 20 September 2026 (bukan hasil Colab Drive)

11 tes lulus. Empat PCAP menghasilkan 29.838 kandidat, 2.068 kecocokan unik dalam
pilot, 27.770 tidak cocok. **Seluruh 2.068 kecocokan panjangnya satu paket**:
Facetime 2.000, Skype 65, BitTorrent 3. Belum ada kecocokan multi-paket untuk IAT.
Tinba mencapai batas 10.000 sesi, sehingga hasilnya parsial. Hasil ini tidak
membuktikan seluruh dataset gagal; batas cakupan dan kebijakan sesi masih berlaku.
Colab memverifikasi sumber Drive dan mengeluarkan angka sendiri. Tahap berikutnya
ditentukan oleh ketersediaan sequence multi-paket asli, bukan sekadar match count.
