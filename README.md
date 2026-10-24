# 🏆 BDC Statistics Explore 2026: Klasifikasi Citra Bencana

**Juara 1** di [Big Data Competition Statistics Explore 2026](https://www.kaggle.com/competitions/big-data-competition-statistics-explore-2026/leaderboard), jalur citra.

| Leaderboard | Micro F1 | Peringkat |
| ----------- | -------- | --------- |
| Public      | 0.933    | #2        |
| **Private** | **0.904** | **#1**   |

**Tim (Institut Teknologi Bandung):** Riantama Putra · Farrel Athalla Putra · Ayu Nagita Cahyaning Wulan

## Tugas

Setiap foto bencana diberi dua label sekaligus:

- **`jenis`**: Banjir, Gempa Bumi, Kebakaran
- **`kerusakan`**: Ringan, Sedang, Berat

Data latih berisi 17.482 citra (9 kelas jenis × kerusakan), data uji 450 citra. Metriknya micro F1 atas 900 baris, yang sama dengan rata-rata akurasi `jenis` dan akurasi `kerusakan`. Karena `jenis` sudah ~99,6%, poin tambahan hanya bisa didapat dari `kerusakan`.

## Solusi

**Backbone:** DINOv3 ViT-7B/16 (`facebook/dinov3-vit7b16-pretrain-lvd1689m`) pada input 384px dan bf16. Hanya 6 blok terakhir dari 40 yang dilatih (1,0 miliar dari 6,7 miliar parameter).

**1. Validasi tanpa bocor.** Sebanyak 38% citra punya kembaran yang nyaris identik, kebanyakan dari kamera sungai yang dipasang tetap. Kami mengelompokkan adegan dengan *difference hash* lalu menambahkan pencocokan yang tahan rotasi dan cermin (deskriptor polar-Fourier dengan verifikasi NCC). Setelah itu dipakai `StratifiedGroupKFold`. Split acak biasa akan membocorkan **45%** baris validasi.

**2. Kepala khusus kerusakan.** Bukti kerusakan bersifat lokal, berurutan, dan maknanya beda untuk tiap jenis bencana.

| Masalah | Solusi |
| ------- | ------ |
| *Mean pooling* menyamarkan kerusakan lokal | 4 aliran fitur: CLS, mean, *gated attention pool*, top-10% patch |
| Arti "berat" beda untuk tiap jenis bencana | **FiLM**: fitur kerusakan dikondisikan pada probabilitas jenis |
| Ringan < Sedang < Berat itu berurutan | Kepala ordinal **CORN** + target lunak **SORD** |
| Interaksi jenis × kerusakan | Kepala gabungan 9 kelas |

**3. Label berderau.** Sekitar 19% citra berada di kelompok adegan yang labelnya saling bertentangan, dan 91% konfliknya hanya selisih satu tingkat. Karena itu dipakai target lunak dan *label smoothing*, bukan one-hot.

**4. Augmentasi yang menjaga bukti.** MixUp tanpa CutMix, karena CutMix bisa memotong area kerusakan. *Hue jitter* dibatasi ±0,02 karena warna adalah sinyal utama untuk `jenis`. Kami juga menambahkan derau Gaussian, sebab satu kelas punya derau 10× lebih tinggi dari kelas lain dan ini bisa jadi jalan pintas.

**5. Dekode.** Bobot paduan kepala dan *offset* log-prob per kelas disetel di validasi untuk mengurangi bias ke kelas "Sedang".

## Hasil validasi (fold 0, berkelompok)

| Akurasi jenis | Akurasi kerusakan | Micro F1 |
| ------------- | ----------------- | -------- |
| 0.9963        | 0.7812            | 0.8888   |

Batas atas teoretis akurasi kerusakan (label mayoritas per adegan) adalah 0,927.

## Struktur repo

```
notebooks/04_dinov3_severity_enhanced.ipynb   # pipeline lengkap: EDA → validasi → training → submisi
results/submission_dinov3_vit7b16_severity.csv  # submisi final
docs/paper.pdf                                  # paper lomba
```

## Menjalankan

- **Kebutuhan:** GPU 80 GB (A100/H100), token Hugging Face yang sudah disetujui lisensi DINOv3, dan kredensial Kaggle.
- **Training:** sekitar 2,8 jam untuk 8 epoch di A100 80GB. Isi `HF_TOKEN` dan `KAGGLE_API_TOKEN` di sel *Kredensial* (atau lewat env var), lalu Run All.
- **Inferensi saja:** set `RUN_MODE = 'INFERENCE'` dan arahkan `MODEL_PATH` ke bobot `dinov3_vit7b16_severity.pt`. Bobot ini berukuran 2 GB sehingga tidak disimpan di repo.

Format submisi: `jenis` 1/2/3 = Banjir/Gempa Bumi/Kebakaran, `kerusakan` 1/2/3 = Berat/Ringan/Sedang.
