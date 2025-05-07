import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import Pipeline # Tidak digunakan secara langsung di app.py setelah load
# from sklearn.preprocessing import LabelEncoder # Tidak digunakan secara langsung di app.py setelah load

# --- Define Custom Transformers (MUST be the same as in training script) ---
# Ensure these classes are identical to those in your training script
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # New features - Ensure column names match your training data
        X['avg_grade'] = (X['Curricular_units_1st_sem_grade'] + X['Curricular_units_2nd_sem_grade']) / 2
        X['avg_approved'] = (X['Curricular_units_1st_sem_approved'] + X['Curricular_units_2nd_sem_approved']) / 2
        # Use .div() and fillna for division by zero/NaNs
        X['approval_rate_1st'] = X['Curricular_units_1st_sem_approved'].div(X['Curricular_units_1st_sem_enrolled'].replace(0, np.nan)).fillna(0)
        X['approval_rate_2nd'] = X['Curricular_units_2nd_sem_approved'].div(X['Curricular_units_2nd_sem_enrolled'].replace(0, np.nan)).fillna(0)
        X['parental_education_avg'] = (X['Fathers_qualification'] + X['Mothers_qualification']) / 2
        X['parental_occupation_avg'] = (X['Fathers_occupation'] + X['Mothers_occupation']) / 2
        X['low_income_flag'] = ((X['Scholarship_holder'] == 1) & (X['Tuition_fees_up_to_date'] == 0)).astype(int)
        X['foreign_and_displaced'] = ((X['International'] == 1) | (X['Displaced'] == 1)).astype(int)
        X['no_eval_first_sem'] = (X['Curricular_units_1st_sem_without_evaluations'] > 0).astype(int)
        X['no_eval_second_sem'] = (X['Curricular_units_2nd_sem_without_evaluations'] > 0).astype(int)

        # Handle NaN values by filling with 0 (as in training)
        X.fillna(0, inplace=True)

        return X

class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols_existing = [col for col in self.cols_to_drop if col in X.columns]
        return X.drop(columns=cols_existing)


# --- Load the trained model and label encoder ---
try:
    pipeline = joblib.load('random_forest_pipeline.pkl')
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please run the training script first to generate 'random_forest_pipeline.pkl' and 'label_encoder.pkl'.")
    st.stop() # Stop the app if files are not found
except Exception as e:
    st.error(f"An error occurred while loading model files: {e}")
    st.stop()

# --- Streamlit App Interface ---
st.set_page_config(layout="wide") # Use wide layout for more space
st.title("Prediksi Status Mahasiswa (Dropout/Lulus)")
st.write("Masukkan detail mahasiswa untuk memprediksi apakah mereka akan menjadi Dropout atau Lulus.")

# Define the list of all original feature columns
original_feature_cols = [
    'Marital_status', 'Application_mode', 'Application_order', 'Course',
    'Daytime_evening_attendance', 'Previous_qualification',
    'Previous_qualification_grade', 'Nacionality', 'Mothers_qualification',
    'Fathers_qualification', 'Mothers_occupation', 'Fathers_occupation',
    'Admission_grade', 'Displaced', 'Educational_special_needs', 'Debtor',
    'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
    'Age_at_enrollment', 'International',
    'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_without_evaluations', 'Unemployment_rate',
    'Inflation_rate', 'GDP'
]

# Create input widgets for each feature
input_values = {}

st.header("Informasi Pribadi & Aplikasi Mahasiswa")
col1, col2 = st.columns(2)
with col1:
    input_values['Marital_status'] = st.number_input(
        "Status Pernikahan", min_value=1, max_value=6, value=1,
        help="Status pernikahan mahasiswa.\n1: Lajang, 2: Menikah, 3: Duda/Janda, 4: Cerai, 5: Hidup Bersama, 6: Terpisah Secara Hukum"
    )
    input_values['Application_mode'] = st.number_input(
        "Mode Aplikasi", min_value=1, value=1,
        help="Metode aplikasi yang digunakan.\nContoh:\n1: Fase 1 - kontingen umum\n7: Pemegang ijazah kursus tinggi lainnya\n15: Mahasiswa internasional (sarjana)\n39: Di atas 23 tahun\n42: Transfer\n43: Pindah jurusan\nLainnya: Lihat dokumentasi untuk kode lengkap."
    )
    input_values['Application_order'] = st.number_input(
        "Urutan Aplikasi", min_value=0, max_value=9, value=1,
        help="Urutan pilihan aplikasi (0: pilihan pertama, 9: pilihan terakhir)."
    )
    input_values['Course'] = st.number_input(
        "Kode Program Studi", min_value=0, value=9119,
        help="Kode program studi yang diambil.\nContoh:\n33: Teknologi Produksi Biofuel\n171: Desain Animasi dan Multimedia\n9119: Teknik Informatika\n9500: Keperawatan\nLainnya: Lihat dokumentasi untuk kode lengkap."
    )
    input_values['Daytime_evening_attendance'] = st.selectbox(
        "Waktu Perkuliahan", options=[1, 0],
        format_func=lambda x: 'Siang (Daytime)' if x == 1 else 'Malam (Evening)',
        help="Apakah mahasiswa menghadiri kelas di siang atau malam hari.\n1: Siang, 0: Malam"
    )
    input_values['Nacionality'] = st.number_input(
        "Kewarganegaraan", min_value=1, value=1,
        help="Kewarganegaraan mahasiswa.\nContoh:\n1: Portugis\n21: Angola\n41: Brazil\nLainnya: Lihat dokumentasi untuk kode lengkap."
    )
    input_values['Age_at_enrollment'] = st.number_input(
        "Usia Saat Pendaftaran", min_value=15, max_value=70, value=18,
        help="Usia mahasiswa pada saat pendaftaran."
    )

with col2:
    input_values['Previous_qualification'] = st.number_input(
        "Kualifikasi Sebelumnya", min_value=1, value=1,
        help="Kualifikasi yang diperoleh sebelum mendaftar di pendidikan tinggi.\nContoh:\n1: Pendidikan menengah\n2: Pendidikan tinggi - Sarjana (S1)\n39: Kursus spesialisasi teknologi\nLainnya: Lihat dokumentasi untuk kode lengkap."
    )
    input_values['Previous_qualification_grade'] = st.number_input(
        "Nilai Kualifikasi Sebelumnya", min_value=0.0, max_value=200.0, value=120.0,
        help="Nilai kualifikasi sebelumnya (antara 0 dan 200)."
    )
    input_values['Admission_grade'] = st.number_input(
        "Nilai Penerimaan", min_value=0.0, max_value=200.0, value=120.0,
        help="Nilai penerimaan mahasiswa (antara 0 dan 200)."
    )
    input_values['Gender'] = st.selectbox(
        "Jenis Kelamin", options=[1, 0],
        format_func=lambda x: 'Laki-laki' if x == 1 else 'Perempuan', # Disesuaikan dengan deskripsi tabel: 1 – male 0 – female
        help="Jenis kelamin mahasiswa.\n1: Laki-laki, 0: Perempuan"
    )
    input_values['International'] = st.selectbox(
        "Mahasiswa Internasional", options=[0, 1],
        format_func=lambda x: 'Ya' if x == 1 else 'Tidak',
        help="Apakah mahasiswa tersebut adalah mahasiswa internasional.\n1: Ya, 0: Tidak"
    )
    input_values['Displaced'] = st.selectbox(
        "Mahasiswa Pindahan/Displaced", options=[0, 1],
        format_func=lambda x: 'Ya' if x == 1 else 'Tidak',
        help="Apakah mahasiswa tersebut adalah orang yang dipindahkan (displaced).\n1: Ya, 0: Tidak"
    )
    input_values['Educational_special_needs'] = st.selectbox(
        "Kebutuhan Pendidikan Khusus", options=[0, 1],
        format_func=lambda x: 'Ya' if x == 1 else 'Tidak',
        help="Apakah mahasiswa memiliki kebutuhan pendidikan khusus.\n1: Ya, 0: Tidak"
    )


st.header("Informasi Orang Tua")
col_parents1, col_parents2 = st.columns(2)
with col_parents1:
    input_values['Mothers_qualification'] = st.number_input(
        "Kualifikasi Ibu", min_value=1, value=1,
        help="Kualifikasi pendidikan ibu.\nContoh:\n1: Pendidikan Menengah\n2: Sarjana (S1)\n37: Pendidikan Dasar Siklus 1 (SD kelas 4/5)\nLainnya: Lihat dokumentasi untuk kode lengkap."
    )
    input_values['Mothers_occupation'] = st.number_input(
        "Pekerjaan Ibu", min_value=0, value=5,
        help="Pekerjaan ibu.\n0: Pelajar\n1: Pejabat Legislatif/Eksekutif, Direktur\n5: Pekerja Jasa Pribadi, Keamanan, Penjual\n9: Pekerja Tidak Terampil\nLainnya: Lihat dokumentasi untuk kode lengkap."
    )
with col_parents2:
    input_values['Fathers_qualification'] = st.number_input(
        "Kualifikasi Ayah", min_value=1, value=1,
        help="Kualifikasi pendidikan ayah.\nContoh:\n1: Pendidikan Menengah\n2: Sarjana (S1)\n37: Pendidikan Dasar Siklus 1 (SD kelas 4/5)\nLainnya: Lihat dokumentasi untuk kode lengkap."
    )
    input_values['Fathers_occupation'] = st.number_input(
        "Pekerjaan Ayah", min_value=0, value=5,
        help="Pekerjaan ayah.\n0: Pelajar\n1: Pejabat Legislatif/Eksekutif, Direktur\n5: Pekerja Jasa Pribadi, Keamanan, Penjual\n9: Pekerja Tidak Terampil\nLainnya: Lihat dokumentasi untuk kode lengkap."
    )


st.header("Status Finansial & Beasiswa")
col_fin1, col_fin2 = st.columns(2)
with col_fin1:
    input_values['Debtor'] = st.selectbox(
        "Memiliki Tunggakan", options=[0, 1],
        format_func=lambda x: 'Ya' if x == 1 else 'Tidak',
        help="Apakah mahasiswa memiliki tunggakan pembayaran.\n1: Ya, 0: Tidak"
    )
    input_values['Tuition_fees_up_to_date'] = st.selectbox(
        "Biaya Kuliah Lunas", options=[1, 0], # Sesuai deskripsi, 1 yes, 0 no
        format_func=lambda x: 'Ya' if x == 1 else 'Tidak',
        help="Apakah biaya kuliah mahasiswa sudah lunas/terbaru.\n1: Ya, 0: Tidak"
    )
with col_fin2:
    input_values['Scholarship_holder'] = st.selectbox(
        "Penerima Beasiswa", options=[0, 1],
        format_func=lambda x: 'Ya' if x == 1 else 'Tidak',
        help="Apakah mahasiswa adalah penerima beasiswa.\n1: Ya, 0: Tidak"
    )


st.header("Kinerja Akademik (Semester 1)")
col5, col6 = st.columns(2)
with col5:
    input_values['Curricular_units_1st_sem_credited'] = st.number_input(
        "SKS Diakui Sem 1", min_value=0, value=0,
        help="Jumlah satuan kredit semester (SKS) yang diakui/dikreditkan di semester pertama."
    )
    input_values['Curricular_units_1st_sem_enrolled'] = st.number_input(
        "SKS Diambil Sem 1", min_value=0, value=6,
        help="Jumlah SKS yang diambil/terdaftar di semester pertama."
    )
    input_values['Curricular_units_1st_sem_evaluations'] = st.number_input(
        "SKS Dievaluasi Sem 1", min_value=0, value=6,
        help="Jumlah SKS yang dievaluasi di semester pertama."
    )
with col6:
    input_values['Curricular_units_1st_sem_approved'] = st.number_input(
        "SKS Lulus Sem 1", min_value=0, value=6,
        help="Jumlah SKS yang berhasil dilulusi di semester pertama."
    )
    input_values['Curricular_units_1st_sem_grade'] = st.number_input(
        "Rata-rata Nilai Sem 1", min_value=0.0, max_value=20.0, value=14.0, # Asumsi skala 0-20, sesuaikan jika beda
        help="Rata-rata nilai mahasiswa untuk semester pertama."
    )
    input_values['Curricular_units_1st_sem_without_evaluations'] = st.number_input(
        "SKS Tanpa Evaluasi Sem 1", min_value=0, value=0,
        help="Jumlah SKS tanpa evaluasi di semester pertama."
    )

st.header("Kinerja Akademik (Semester 2)")
col7, col8 = st.columns(2)
with col7:
    input_values['Curricular_units_2nd_sem_credited'] = st.number_input(
        "SKS Diakui Sem 2", min_value=0, value=0,
        help="Jumlah SKS yang diakui/dikreditkan di semester kedua."
    )
    input_values['Curricular_units_2nd_sem_enrolled'] = st.number_input(
        "SKS Diambil Sem 2", min_value=0, value=6,
        help="Jumlah SKS yang diambil/terdaftar di semester kedua."
    )
    input_values['Curricular_units_2nd_sem_evaluations'] = st.number_input(
        "SKS Dievaluasi Sem 2", min_value=0, value=6,
        help="Jumlah SKS yang dievaluasi di semester kedua."
    )
with col8:
    input_values['Curricular_units_2nd_sem_approved'] = st.number_input(
        "SKS Lulus Sem 2", min_value=0, value=6,
        help="Jumlah SKS yang berhasil dilulusi di semester kedua."
    )
    input_values['Curricular_units_2nd_sem_grade'] = st.number_input(
        "Rata-rata Nilai Sem 2", min_value=0.0, max_value=20.0, value=14.0, # Asumsi skala 0-20, sesuaikan jika beda
        help="Rata-rata nilai mahasiswa untuk semester kedua."
    )
    input_values['Curricular_units_2nd_sem_without_evaluations'] = st.number_input(
        "SKS Tanpa Evaluasi Sem 2", min_value=0, value=0,
        help="Jumlah SKS tanpa evaluasi di semester kedua."
    )

st.header("Faktor Makroekonomi")
col9, col10, col11 = st.columns(3) # Menggunakan 3 kolom agar lebih rapi
with col9:
    input_values['Unemployment_rate'] = st.number_input(
        "Tingkat Pengangguran (%)", value=10.0, step=0.1,
        help="Tingkat pengangguran saat pendaftaran mahasiswa."
    )
with col10:
    input_values['Inflation_rate'] = st.number_input(
        "Tingkat Inflasi (%)", value=1.0, step=0.1,
        help="Tingkat inflasi saat pendaftaran mahasiswa."
    )
with col11:
    input_values['GDP'] = st.number_input(
        "PDB (GDP)", value=0.0, step=0.1,
        help="Produk Domestik Bruto (GDP) saat pendaftaran mahasiswa."
    )


# --- Prediction ---
if st.button("Prediksi Status", type="primary", use_container_width=True):
    # Create a DataFrame from input values with the EXACT original column order
    try:
        input_data = pd.DataFrame([input_values], columns=original_feature_cols)

        # Make prediction using the pipeline
        prediction_numerical = pipeline.predict(input_data)

        # Decode the numerical prediction back to the original label
        prediction_label = le.inverse_transform(prediction_numerical)

        st.subheader("Hasil Prediksi:")

        # Ambil kode numerik untuk 'Dropout' dan 'Graduate' dari label encoder
        # Pastikan string 'Dropout' dan 'Graduate' ada di le.classes_
        dropout_code = -1
        graduate_code = -1

        # Cari kode numerik berdasarkan kelas yang diketahui oleh LabelEncoder
        if 'Dropout' in le.classes_:
            dropout_code = le.transform(['Dropout'])[0]
        if 'Graduate' in le.classes_:
            graduate_code = le.transform(['Graduate'])[0]
        
        predicted_status_text = ""
        if prediction_numerical[0] == dropout_code:
            predicted_status_text = "Dropout"
            st.error(f"Prediksi Status: **{predicted_status_text}** 😥")
        elif prediction_numerical[0] == graduate_code:
            predicted_status_text = "Lulus (Graduate)"
            st.success(f"Prediksi Status: **{predicted_status_text}** 🎉")
        else:
            # Fallback jika label tidak dikenali (seharusnya tidak terjadi jika le.classes_ benar)
            predicted_status_text = f"Status Tidak Diketahui (Kode: {prediction_numerical[0]})"
            st.warning(f"Prediksi Status: **{prediction_label[0]}** (Label asli dari model)")
            st.write(f"Kode numerik yang dihasilkan: {prediction_numerical[0]}")
            st.write(f"Label encoder classes: {le.classes_}")
            st.write(f"Mapping yang diharapkan: Dropout -> {dropout_code}, Graduate -> {graduate_code}")


    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.error("Pastikan semua input diisi dengan benar dan model telah dilatih dengan fitur yang sesuai.")


st.markdown("---")
st.caption("Catatan: Ini adalah model prediktif dan hasilnya harus diinterpretasikan dengan hati-hati.")
st.caption("Pastikan nilai input akurat dan mencerminkan skala serta makna yang sama dengan data pelatihan.")
