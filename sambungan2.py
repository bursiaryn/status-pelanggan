import pickle
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score

# Membaca model
customer_model = pickle.load(open('pelanggan_model.sav', 'rb'))

# Judul web
st.title('Prediksi Status Pelanggan')

# Input data dengan label yang sesuai
CreditScore = st.text_input('Skor Kredit Pelanggan')
Gender = st.text_input('Jenis Kelamin (0: Wanita, 1: Pria)')
Age = st.text_input('Umur')
Tenure = st.text_input('Masa Kerja (tahun)')
Balance = st.text_input('Saldo')
NumOfProducts = st.text_input('Jumlah Produk yang Digunakan')
HasCrCard = st.text_input('Memiliki Kartu Kredit (0: Tidak, 1: Ya)')
IsActiveMember = st.text_input('Anggota Aktif (0: Tidak, 1: Ya)')
EstimatedSalary = st.text_input('Perkiraan Gaji')

Prediksi_Status_Pelanggan = ''

# Membuat tombol untuk prediksi
if st.button('Prediksi'):
    try:
        # Konversi input menjadi numerik
        inputs = np.array([[float(CreditScore), float(Gender), float(Age), 
                            float(Tenure), float(Balance), float(NumOfProducts), 
                            float(HasCrCard), float(IsActiveMember), float(EstimatedSalary)]])
        
        # Lakukan prediksi
        status_prediksi = customer_model.predict(inputs)
        
        if status_prediksi[0] == 0:
            st.success('Pelanggan Tidak Keluar dari Bank')
        else:
            st.success('Pelanggan Keluar dari Bank')
    except ValueError:
        st.error("Pastikan semua input diisi dengan angka yang valid.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# Contoh visualisasi dan evaluasi model
st.subheader('Evaluasi Model')

# Load data evaluation
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
y_pred = customer_model.predict(x_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:")
st.write(cm)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, customer_model.predict_proba(x_test)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
st.pyplot(plt)

# Laporan Klasifikasi
report = classification_report(y_test, y_pred, output_dict=True)
st.write("Laporan Klasifikasi:")
st.write(report)
