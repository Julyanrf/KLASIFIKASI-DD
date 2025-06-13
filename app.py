import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# === Load model dan data training ===
model = joblib.load('model_svm.pkl')
df_final = joblib.load('df_final.pkl')

# === Fungsi kategorisasi ===
def kategori_trombosit(x):
    if x < 100000:
        return 1
    elif x <= 150000:
        return 2
    else:
        return 3

def kategori_hemoglobin(x):
    if x < 12:
        return 1
    elif x <= 16.9:
        return 2
    else:
        return 3

def kategori_hematokrit(x):
    if x < 35:
        return 1
    elif x <= 49.9:
        return 2
    else:
        return 3

# Pastikan kolom 'Jenis_kelamin' ada
if 'Jenis Kelamin' in df_final.columns and 'Jenis_kelamin' not in df_final.columns:
    label_map_gender = {'l': 1, 'p': 2, 'laki-laki': 1, 'perempuan': 2}
    df_final['Jenis_kelamin'] = df_final['Jenis Kelamin'].apply(lambda x: label_map_gender.get(str(x).lower().strip(), -1))

fit_columns = [
    'NO', 'Umur', 'Demam', 'Pendarahan', 'Pusing', 'Nyeri Otot/Sendi',
    'Trombosit', 'Hemoglobin', 'Hematokrit',
    'Trombosit_Kat', 'Hemoglobin_Kat', 'Hematokrit_Kat', 'Jenis_kelamin'
]

actual_fit_columns = [col for col in fit_columns if col in df_final.columns]

diagnosis_map = {1: "DD (Demam Dengue)", 2: "DBD (Demam Berdarah Dengue)", 3: "DSS (Sindrom Syok Dengue)"}

scaler = StandardScaler()
scaler.fit(df_final[actual_fit_columns])

# === Aplikasi Streamlit ===
st.title("Prediksi Diagnosis Demam Berdarah Menggunakan SVM")

with st.form("form_pasien"):
    No = st.number_input("No", step=1, format="%d")
    Nama = st.text_input("Nama")
    Umur = st.number_input("Umur", step=1.0)
    Jenis_Kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])

    Demam = st.selectbox("Demam", ["YA", "TIDAK"])
    Pendarahan = st.selectbox("Pendarahan", ["YA", "TIDAK"])
    Pusing = st.selectbox("Pusing", ["YA", "TIDAK"])
    Nyeri = st.selectbox("Nyeri Otot/Sendi", ["YA", "TIDAK"])

    Trombosit = st.number_input("Trombosit", step=1000.0, format="%f")
    Hemoglobin = st.text_input("Hemoglobin (contoh: 13.5)")
    Hematokrit = st.text_input("Hematokrit (contoh: 40.0)")

    # Validasi dasar agar tidak tertukar
    # This block is now correctly indented within the form
    if Hemoglobin and Hematokrit: # Ensure inputs are not empty before attempting conversion/comparison
        try:
            # Convert to float here for immediate comparison if possible
            # Or, handle the string comparison logic if preferred
            hemo_float = float(Hemoglobin.replace(",", "."))
            hemato_float = float(Hematokrit.replace(",", "."))


    submitted = st.form_submit_button("Prediksi")


        Demam_val = 1 if Demam.lower() in ['ya', '1'] else 0
        Pendarahan_val = 1 if Pendarahan.lower() in ['ya', '1'] else 0
        Pusing_val = 1 if Pusing.lower() in ['ya', '1'] else 0
        Nyeri_val = 1 if Nyeri.lower() in ['ya', '1'] else 0

        Trombosit_Kat = kategori_trombosit(Trombosit)
        Hemoglobin_Kat = kategori_hemoglobin(Hemoglobin_float) # Use the converted float value
        Hematokrit_Kat = kategori_hematokrit(Hematokrit_float) # Use the converted float value

        label_jenis_kelamin_input = {'laki-laki': 1, 'perempuan': 2}
        Jenis_Kelamin_encoded = label_jenis_kelamin_input.get(Jenis_Kelamin.lower(), -1)

        input_data = pd.DataFrame([[
            No, Umur, Demam_val, Pendarahan_val, Pusing_val, Nyeri_val,
            Trombosit, Hemoglobin_float, Hematokrit_float, # Use float values in DataFrame
            Trombosit_Kat, Hemoglobin_Kat, Hematokrit_Kat,
            Jenis_Kelamin_encoded
        ]], columns=fit_columns)

        input_data_scaled = input_data[actual_fit_columns]

        if input_data_scaled.isnull().values.any():
            st.error("Terdapat nilai kosong dalam data input.")
        else:
            input_scaled = scaler.transform(input_data_scaled)
            prediction = model.predict(input_scaled)
            diagnosis = diagnosis_map.get(prediction[0], "Diagnosis tidak ditemukan")

            st.success(f"Nama: {Nama}")
            st.success(f"Hasil Prediksi: {diagnosis}")

    except ValueError:
        st.error("Pastikan nilai Hemoglobin dan Hematokrit diisi dengan angka.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
