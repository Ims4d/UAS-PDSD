# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# Set konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Analisis Data Pengiriman Pesanan",
    layout="wide"
)

def page_1():
    st.title("Beranda")

def page_2():
    st.title("Rata-rata Waktu Pengiriman Pesanan")

def page_3():
    st.title("Persentase dan Estimasi Keterlambatan Pengiriman")

def page_4():
    st.title("Distribusi Status Pengiriman")

def page_5():
    st.title("Rata-rata Waktu Antara Pesanan Dibuat Hingga Disetujui")

def page_6():
    st.title("Pesanan Yang Berhasil Dikirim Dalam Satu Bulan Terakhir")

def page_7():
    st.title("Perbedaan Waktu Pengiriman Berdasarkan Waktu Pembelian")

def page_8():
    st.title("Informasi Kelompok")

# Sidebar Navigasi
st.sidebar.title("Navigasi Dashboard")
pg = st.navigation([
    st.Page(page_1, title="Beranda"),
    st.Page(page_2, title="Rata-rata Waktu Pengiriman Pesanan"),
    st.Page(page_3, title="Persentase dan Estimasi Keterlambatan Pengiriman"),
    st.Page(page_4, title="Distribusi Status Pengiriman"),
    st.Page(page_5, title="Rata-rata Waktu Antara Pesanan Dibuat Hingga Disetujui"),
    st.Page(page_6, title="Pesanan Yang Berhasil Dikirim Dalam Satu Bulan Terakhir"),
    st.Page(page_7, title="Perbedaan Waktu Pengiriman Berdasarkan Waktu Pembelian"),
    st.Page(page_8, title="Informasi Kelompok"),
])
pg.run()
st.sidebar.markdown("---")
st.sidebar.markdown("Kelompok 2 PDSD | FTIK UNIKOM | 2025")

# Konten Halaman
# if menu == "Beranda":
#     st.title("Dashboard Analisis Data Pengiriman Pesanan")
#     st.image("https://via.placeholder.com/800x300", use_container_width=True)
#     st.markdown(
#         """
#         Dashboard ini menyajikan berbagai analisis terkait pengiriman pesanan.
#         Silakan pilih menu di sebelah kiri untuk melihat detail analisis.
#         """
#     )

# elif menu == "Analisis Data":
#     st.title("Analisis Data Pengiriman")
    
#     @st.cache_data
#     def load_data():
#         url = 'https://raw.githubusercontent.com/Ims4d/sementara/refs/heads/main/orders_dataset.csv'
#         df = pd.read_csv(url)
#         df.fillna({
#             'order_approved_at': pd.Timestamp.max, 
#             'order_delivered_carrier_date': pd.Timestamp.max, 
#             'order_delivered_customer_date': pd.Timestamp.max
#         }, inplace=True)
#         date_cols = ['order_purchase_timestamp', 'order_approved_at', 
#                      'order_delivered_carrier_date', 'order_delivered_customer_date', 
#                      'order_estimated_delivery_date']
#         for col in date_cols:
#             df[col] = pd.to_datetime(df[col])
#         df['days_to_carrier'] = (df['order_delivered_carrier_date'] - df['order_purchase_timestamp']).dt.days
#         return df
    
#     df = load_data()
    
#     st.subheader("📌 Informasi Dataset")
#     col1, col2 = st.columns(2)
#     col1.metric("📊 Total Data", f"{df.shape[0]} baris")
#     col2.metric("📋 Total Kolom", f"{df.shape[1]}")
    
#     option = st.selectbox("Pilih Pertanyaan Bisnis:", 
#                           ["Rata-rata Waktu Pengiriman ke Kurir", 
#                            "Persentase Pesanan Tepat Waktu vs Terlambat", 
#                            "Distribusi Status Pengiriman"])
    
#     if option == "Rata-rata Waktu Pengiriman ke Kurir":
#         st.subheader("📦 Rata-rata Waktu Pengiriman ke Kurir")
#         avg_time = df['days_to_carrier'].mean()
#         st.markdown(f"### Rata-rata waktu pengiriman ke kurir: **{avg_time:.0f} hari**")
#         fig, ax = plt.subplots(figsize=(10, 6))
#         sns.histplot(df['days_to_carrier'].dropna(), bins=20, kde=True, color='blue')
#         ax.set_title('Distribusi Waktu Pengiriman ke Kurir')
#         ax.set_xlabel('Hari')
#         ax.set_ylabel('Frekuensi')
#         st.pyplot(fig)
    
# elif menu == "Tentang Kelompok":
#     st.title("📌 Tentang Kelompok")
#     st.image("https://via.placeholder.com/800x400", use_column_width=True)
#     st.markdown(
#         """
#         **Kelompok 2 - Analisis Data Pengiriman Pesanan**
        
#         🔹 **Muhamad Irsad Assopi (10123038)**  
#         🔹 **Yoan Ready Syavera (10123015)**  
#         🔹 **Rizky Al Farid Hafizh (10123028)**  
#         🔹 **Hizkia Imanuel Edho (10123020)**  
#         🔹 **Nur Ain Salimah (10123049)**  
#         🔹 **Wa Ode Syahwa Salsabilah (10123062)**  
        
#         Proyek ini dibuat sebagai bagian dari analisis data untuk memahami pola pengiriman pesanan.
#         """
#     )
