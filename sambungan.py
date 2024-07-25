import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler

# Membaca model
customer_model = pickle.load(open('pelanggan_model.sav','rb'))

# Judul web
st.title('Prediksi Status Pelanggan')

# Input data dengan contoh angka valid untuk pengujian
CreditScore = st.text_input('Uang kredit pelanggan')
Gender = st.text_input('Suhu Maksimum')
Age = st.text_input('Suhu Minimum')
Tenure = st.text_input('Kecepatan Angin')
Balance = st.text_input('Uang kredit pelanggan')
NumOfProducts = st.text_input('Suhu Maksimum')
HasCrCard = st.text_input('Suhu Minimum')
IsActiveMember = st.text_input('Kecepatan Angin')
EstimatedSalary = st.text_input('Kecepatan Angin')

Prediksi_Status_Pelanggan = ''

# Membuat tombol untuk prediksi
if st.button('Prediksi'):
    try:
        # Konversi input menjadi numerik
        inputs = np.array([[float(CreditScore), float(Gender), float(Age), 
                            float(Tenure),float(Balance), float(NumOfProducts), float(HasCrCard), 
                            float(IsActiveMember), float(EstimatedSalary)]])
        # Lakukan prediksi
        status_prediksi = customer_model.predict(inputs)
        
        if status_prediksi[0] == 0:
            status_prediksi = 'Pelanggan Tidak Keluar dari Bank'
            st.success(status_prediksi)
        if status_prediksi[0] == 1:
            status_prediksi = 'Pelanggan Keluar dari Bank'
            st.success(status_prediksi)
    except ValueError:
        st.error("Pastikan semua input diisi dengan angka yang valid.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
