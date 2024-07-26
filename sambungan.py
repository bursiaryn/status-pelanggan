import pickle
import streamlit as st
import numpy as np

# Membaca model
file_path = 'pelanggan_model.sav'

# Membaca file dan memuat data
with open(file_path, 'rb') as file:
    data = pickle.load(file)


# Judul web
st.title('Prediksi Status Pelanggan')

# Input data dengan contoh angka valid untuk pengujian
CreditScore = st.text_input('Uang kredit pelanggan', key= 'CreditScore')
Gender = st.selectbox('Jenis kelamin', options=[(0, 'Perempuan'), (1, 'Laki-laki')], format_func=lambda x: x[1],key = 'Gender')
Age = st.text_input('Umur',key = 'Age')
Tenure = st.text_input('Masa Jabatan',key = 'Tenure')
Balance = st.text_input('Uang kredit pelanggan',key = 'Balance')
NumOfProducts = st.selectbox('Jumlah Produk Bank', options=[1, 2, 3, 4],key = 'NumOfProducts')
HasCrCard = st.selectbox('Kepunyaan Kartu Kredit', options=[(0, 'Tidak'), (1, 'Ya')],format_func=lambda x: x[1],key = 'HasCrCard')
IsActiveMember = st.selectbox('Keaktifan Member', options=[(0, 'Tidak Aktif'), (1, 'Aktif')], format_func=lambda x: x[1], key = 'IsActiveMember')
EstimatedSalary = st.text_input('Perkiraan Gaji',key='EstimatedSalary')

Prediksi_Status_Pelanggan = ''

# Membuat tombol untuk prediksi
if st.button('Prediksi'):
    try:
        # Konversi input menjadi numerik
        inputs = np.array([[float(CreditScore), Gender[0], float(Age), float(Tenure),float(Balance),float(NumOfProducts), 
                            HasCrCard[0], IsActiveMember[0], float(EstimatedSalary)]])
        # Lakukan prediksi
        status_prediksi = data.predict(inputs)
        
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
