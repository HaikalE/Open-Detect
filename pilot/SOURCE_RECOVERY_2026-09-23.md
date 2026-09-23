# P1 lanjutan: pemulihan urutan paket USTC

Status: **kandidat berpasangan ditemukan; belum siap training/evaluasi tesis**.
Laporan lokal: `pilot/reports/anchored_coverage_summary_20260923.json`.
Folder `reports/` sengaja tidak di-commit; salinan ringkasan diunggah ke folder
pilot Drive. Skrip dan tes reproduksinya ada di branch ini.

## Metode

Setiap citra USTC NPZ dipecah menjadi delapan slot 128 byte. Hash slot pertama
dicari pada paket PCAP yang diserialisasi dengan fungsi legacy Open-Detect.
Seluruh slot terisi berikutnya wajib cocok unik dengan paket dari biflow yang
sama, dalam urutan capture, dengan timestamp sumber. Kandidat ambigu, label
bertentangan, slot hilang, atau multi-paket tanpa IAT positif ditolak. Tidak
ada IAT yang dibuat dari urutan baris NPZ. Pemindaian dibatasi **10.000 paket
fisik pertama per capture**, jadi hitungan ini bukan cakupan seluruh kelas.

| Capture kelas | Citra multi-paket + IAT positif | Flow group | Train NPZ | Test NPZ |
| --- | ---: | ---: | ---: | ---: |
| Cridex | 0 | 0 | 0 | 0 |
| Geodo | 101 | 101 | 89 | 12 |
| Miuref | 187 | 183 | 170 | 17 |
| Tinba | 74 | 72 | 70 | 4 |
| Weibo-1 | 0 | 0 | 0 | 0 |
| WorldOfWarcraft | 151 | 142 | 137 | 14 |
| Zeus | 51 | 51 | 46 | 5 |
| **Total** | **564** | **549** | **512** | **52** |

Audit 564 kandidat: 564 baris NPZ unik, tak ada paket sumber dipakai oleh dua
citra dalam laporan ini, tetapi **tiga flow WorldOfWarcraft menyeberang partisi
train/test NPZ**. Karena itu jangan memakai pembagian NPZ lama secara otomatis
untuk eksperimen temporal. Angka nol pada Cridex/Weibo hanya berlaku untuk
metode serializer legacy dan capture prefix ini; bukan bukti tidak ada pasangan.

## Gerbang lanjut

1. Selesaikan inventaris dan pemindaian capture sumber yang relevan; audit
   duplikat citra/paket lintas semua capture, bukan hanya tujuh laporan ini.
2. Tetapkan unit split berdasarkan flow/capture sebelum membuat dataset paired;
   jangan membiarkan satu flow pada train dan test.
3. Buat image-only control dan image+sequence fusion pada sampel/split **yang
   identik**. Fitur awal: panjang payload, arah, dan IAT dari PCAP; bandingkan
   dengan ablation tanpa IAT. Normalisasi harus fit pada train saja.
4. Tujuh capture ini hanya feasibility pilot, belum mengisi seluruh kelas atau
   skenario A/B/C. Jangan mempresentasikan metriknya sebagai hasil akhir tesis.

Jalankan `python -m pilot.summarize_anchor_coverage <report.json> ...
--output <ringkasan-baru.json>` untuk mengulang audit ringkasan. Skrip
`pilot/recover_anchored_sequences.py` menghasilkan tiap report kelas dari NPZ
USTC dan PCAP sumber.
