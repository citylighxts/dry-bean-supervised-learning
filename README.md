# Supervised Learning pada Dry Bean — Decision Tree vs SVM vs ANN vs CNN

Perbandingan empat metode supervised learning, **Decision Tree**, **Support Vector Machine**,
**Artificial Neural Network (MLP)**, dan **CNN 1D**, untuk klasifikasi 7 varietas kacang kering
pada dataset **Dry Bean** (UCI ML Repository, ID 602) — lengkap dengan EDA, preprocessing,
tuning hyperparameter, evaluasi (Akurasi, Precision, Recall, F1), uji signifikansi, dan analisis
kelebihan/kekurangan tiap metode.

### How to Run

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn torch jupyter
jupyter notebook dry_bean_supervised_learning.ipynb
```

Bisa dijalankan dari folder kosong mana pun, dataset diunduh otomatis dari UCI saat sel pemuatan
data dieksekusi (butuh internet). Seluruh notebook memakan waktu sekitar 9 menit.

## Isi Notebook

| Bagian | Isi |
|---|---|
| Eksplorasi Data (EDA) | Sumber dataset, kamus variabel, statistik deskriptif, distribusi kelas dan fitur, outlier (IQR), korelasi, pemisahan antar kelas |
| Data Cleaning & Preprocessing | Buang duplikat, label encoding, split 80/20 stratified, bukti kenapa standardisasi perlu |
| Implementasi Supervised Learning | Decision Tree, SVM, ANN (`GridSearchCV` 5-fold), CNN 1D (PyTorch, early stopping) |
| Perbandingan Hasil | Tabel metrik, selang kepercayaan bootstrap, F1 per kelas, confusion matrix, uji McNemar, kestabilan 5 seed |
| Eksperimen Perbaikan Model | Fitur rasio untuk Decision Tree, konfigurasi alternatif SVM dan ANN, ablasi resep CNN |
| Analisis dan Kesimpulan | Tabel kelebihan/kekurangan, kriteria pemilihan metode, keterbatasan |

## Ringkasan Hasil

Data uji 2.709 biji (split stratified 80/20, seed 42):

| | Decision Tree | SVM (RBF) | ANN (MLP) | CNN 1D |
|---|---|---|---|---|
| Akurasi | 0,9022 | 0,9236 | 0,9155 | 0,9247 |
| F1 macro | 0,9132 | 0,9347 | 0,9276 | 0,9367 |
| Akurasi, rata-rata 5 seed | – | – | 0,9241 | 0,9230 |
| Waktu latih | ± 0,07 detik | ± 0,1 detik | ± 0,2 detik | puluhan detik |

**Catatan:** SVM, ANN, dan CNN pada dasarnya setara di sekitar 92,3 sampai 92,4% akurasi.
Selisih di antara ketiganya pada satu kali latih lebih banyak dipengaruhi inisialisasi acak
daripada kemampuan metodenya (uji McNemar SVM vs CNN p = 0,79, dan ANN seed 42 ternyata
hasil yang kebetulan rendah). Hanya Decision Tree yang jelas lebih rendah (± 2 poin, signifikan
terhadap ketiga metode lain). Fitur rasio menaikkan Decision Tree ke 0,9121 pada data uji, tapi
kenaikannya di data latih lemah (± 0,001 F1), jadi belum terbukti kuat.

Kesalahan sisa terutama datang dari pasangan varietas yang mirip, `SIRA` dan `DERMASON`, sedangkan
`BOMBAY` selalu terklasifikasi sempurna. Rekomendasi praktis untuk dataset ini adalah **SVM dengan
kernel RBF**: akurasinya setara model terbaik, deterministik, dan paling murah dilatih.

## Tech Stack

- **Python** — pandas, numpy, scipy, scikit-learn, matplotlib, seaborn
- **Pemuatan dataset otomatis** via `urllib.request` + `zipfile` (pustaka standar), berkas ARFF dibaca dengan `scipy.io.arff`
- **Decision Tree, SVM, MLP** dari scikit-learn dengan `Pipeline` dan `GridSearchCV` (F1 macro)
- **CNN 1D** dengan PyTorch (`BatchNorm`, `Dropout`, Adam, early stopping)
- **Evaluasi:** Akurasi, Precision, Recall, F1 (macro dan weighted), confusion matrix, bootstrap CI, uji McNemar

## Catatan Reproduksi

Semua seed dikunci (`RANDOM_STATE = 42`). Bagian scikit-learn seharusnya menghasilkan angka yang sama
di mesin lain dengan versi pustaka yang mirip. Bagian CNN dikunci dengan
`torch.use_deterministic_algorithms(True)` dan 1 thread supaya bisa diulang persis di mesin yang sama,
tapi belum tentu identik di mesin atau versi torch lain.

## Sumber Dataset

Koklu, M. and Ozkan, I. A. (2020). *Dry Bean* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C50S4B — Lisensi CC BY 4.0.
