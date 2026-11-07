# supervised learning pada Dry Bean: decision tree vs SVM vs ANN vs CNN

perbandingan empat metode supervised learning, **Decision Tree**, **support vector machine**,
**artificial neural network (MLP)**, dan **CNN 1D**, buat klasifikasi 7 varietas kacang kering
di dataset **Dry Bean** (UCI ML Repository, ID 602). isinya EDA, preprocessing, tuning
hyperparameter, evaluasi (akurasi, precision, recall, F1), uji signifikansi, dan analisis
kelebihan kekurangan tiap metode.

### how to run

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn torch jupyter
jupyter notebook dry_bean_supervised_learning.ipynb
```

bisa dijalankan dari folder kosong mana pun, dataset diunduh otomatis dari UCI waktu sel pemuatan
data dijalankan (butuh internet). seluruh notebook makan waktu sekitar 9 menit.

## isi notebook

| bagian | isi |
|---|---|
| eksplorasi data (EDA) | sumber dataset, kamus variabel, statistik deskriptif, distribusi kelas dan fitur, outlier (IQR), korelasi, pemisahan antar kelas |
| data cleaning & preprocessing | buang duplikat, label encoding, split 80/20 stratified, bukti kenapa standardisasi perlu |
| implementasi supervised learning | Decision Tree, SVM, ANN (`GridSearchCV` 5-fold), CNN 1D (PyTorch, early stopping) |
| perbandingan hasil | tabel metrik, selang kepercayaan bootstrap, F1 per kelas, confusion matrix, uji McNemar, kestabilan 5 seed |
| eksperimen perbaikan model | fitur rasio buat Decision Tree, konfigurasi alternatif SVM dan ANN, ablasi resep CNN |
| analisis dan kesimpulan | tabel kelebihan kekurangan, kriteria pemilihan metode, keterbatasan |

## ringkasan hasil

data uji 2.709 biji (split stratified 80/20, seed 42):

| | Decision Tree | SVM (RBF) | ANN (MLP) | CNN 1D |
|---|---|---|---|---|
| akurasi | 0,9022 | 0,9236 | 0,9155 | 0,9247 |
| F1 macro | 0,9132 | 0,9347 | 0,9276 | 0,9367 |
| akurasi, rata-rata 5 seed | – | – | 0,9241 | 0,9230 |
| waktu latih | ± 0,07 detik | ± 0,1 detik | ± 0,2 detik | puluhan detik |

**catatan:** SVM, ANN, dan CNN pada dasarnya setara di sekitar 92,3 sampai 92,4% akurasi.
selisih di antara ketiganya pada satu kali latih lebih banyak dipengaruhi inisialisasi acak
daripada kemampuan metodenya (uji McNemar SVM vs CNN p = 0,79, dan ANN seed 42 ternyata
hasil yang kebetulan rendah). cuma Decision Tree yang jelas lebih rendah (± 2 poin, signifikan
terhadap ketiga metode lain). fitur rasio menaikkan Decision Tree ke 0,9121 di data uji, tapi
kenaikannya di data latih lemah (± 0,001 F1), jadi belum terbukti kuat.

kesalahan sisa terutama datang dari pasangan varietas yang mirip, `SIRA` dan `DERMASON`, sedangkan
`BOMBAY` selalu terklasifikasi sempurna. rekomendasi praktis buat dataset ini adalah **SVM dengan
kernel RBF**: akurasinya setara model terbaik, deterministik, dan paling murah dilatih.

## tech stack

- **Python**: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn
- **pemuatan dataset otomatis** lewat `urllib.request` + `zipfile` (pustaka standar), berkas ARFF dibaca pakai `scipy.io.arff`
- **Decision Tree, SVM, MLP** dari scikit-learn dengan `Pipeline` dan `GridSearchCV` (F1 macro)
- **CNN 1D** pakai PyTorch (`BatchNorm`, `Dropout`, Adam, early stopping)
- **evaluasi:** akurasi, precision, recall, F1 (macro dan weighted), confusion matrix, bootstrap CI, uji McNemar

## catatan reproduksi

semua seed dikunci (`randomState = 42`). bagian scikit-learn seharusnya menghasilkan angka yang sama
di mesin lain dengan versi pustaka yang mirip. bagian CNN dikunci pakai
`torch.use_deterministic_algorithms(True)` dan 1 thread supaya bisa diulang persis di mesin yang sama,
tapi belum tentu identik di mesin atau versi torch lain.

## sumber dataset

Koklu, M. and Ozkan, I. A. (2020). *Dry Bean* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C50S4B, lisensi CC BY 4.0.
