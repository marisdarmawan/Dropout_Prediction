
# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini, institusi ini telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout.

Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah besar bagi sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.

### Permasalahan Bisnis

- Tingginya jumlah mahasiswa yang mengalami dropout.
- Belum adanya sistem prediksi untuk mendeteksi risiko dropout.
- Diperlukan sistem yang dapat memberikan informasi awal untuk intervensi.

### Cakupan Proyek

- Menyusun pipeline preprocessing dan model machine learning untuk klasifikasi status mahasiswa.
- Menganalisis data dan menentukan fitur-fitur yang signifikan.
- Membangun prototype aplikasi berbasis Streamlit.
- Membuat business dashboard menggunakan aplikasi Tableau dan diupload ke website.

### Persiapan

Sumber data: Dataset mahasiswa dari Jaya Jaya Institut (4424 baris, 37 kolom) yang mencakup informasi akademik, latar belakang, dan ekonomi.

Setup environment:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

## Business Dashboard

Dashboard visualisasi dibuat menggunakan aplikasi Tableau. Visualisasi yang disediakan meliputi:

- Distribusi status mahasiswa (Dropout, Graduate) dalam bentuk pie chart
- Distribusi status mahasiswa terhadap status perkawinan
- Distribusi status mahasiswa terhadap jam perkuliahan
- Distribusi status mahasiswa terhadap hutang (debt)
- Distribusi status mahasiswa terhadap status pergantian (displace)
- Distribusi status mahasiswa terhadap status pembayaran uang kuliah (tuition)
- Distribusi status mahasiswa terhadap jenis kelamin
- Distribusi status mahasiswa terhadap status beasiswa (scholarship)
- Median umur terhadap status mahasiswa
- Median rate inflasi terhadap status mahasiswa
- Median GDP terhadap status mahasiswa

Visualisasi ini bertujuan untuk memahami lebih lanjut karakteristik mahasiswa dan faktor yang berkontribusi terhadap dropout.
> Link dashboard: https://public.tableau.com/app/profile/mohammad.aris.darmawan/viz/Book1_16761176495170/Dashboard1

## Menjalankan Sistem Machine Learning

Proyek ini membangun pipeline yang terdiri atas:

- Preprocessing (encoding kategori, scaling)
- Model klasifikasi (RandomForestClassifier)
- Evaluasi model (classification report dan akurasi)
- Penyimpanan model (`model.joblib`)
- Deploy aplikasi berbasis Streamlit

Untuk menjalankan aplikasi:
```bash
streamlit run app.py
```

> Link prototipe: https://doprediction-mohammad-aris-darmawan.streamlit.app/

## Conclusion

Sistem prediksi yang dibangun mampu mengklasifikasikan status mahasiswa dengan akurasi yang cukup baik. Model dapat digunakan sebagai alat bantu dalam menentukan intervensi terhadap mahasiswa yang berisiko dropout.

### Rekomendasi Action Items

- Pastikan kualitas pengajaran di semester kedua optimal, karena tingkat kelulusan ditentukan oleh nilai dan banyaknya course yang diambil di semester kedua.
- Perhatikan Status Pembayaran Uang Kuliah: Siswa yang menunggak pembayaran uang kuliah cenderung memiliki risiko dropout yang lebih tinggi. Berikan solusi pembayaran yang fleksibel dan program keringanan bagi mahasiswa yang membutuhkan.
- Berikan layanan seperti rekaman kelas dan sesi konsultasi bagi mahasiswa yang kuliah malam.

