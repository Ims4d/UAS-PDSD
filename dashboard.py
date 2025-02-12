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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Gaya CSS untuk tampilan yang lebih modern
st.markdown(
    """
    <style>
    .css-18e3th9 {
        padding: 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }
    .stSelectbox {
        background-color: #f5f5f5;
    }
    .stSidebar {
        background-color: #263238;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigasi
st.sidebar.image("https://via.placeholder.com/150", use_column_width=True)
st.sidebar.title("Navigasi Dashboard")
menu = st.sidebar.radio("Pilih Menu:", ["Beranda", "Analisis Data", "Tentang Kelompok"])

# Sidebar Informasi Kelompok
st.sidebar.markdown("---")
st.sidebar.subheader("Kelompok 2 - Analisis Pengiriman")
st.sidebar.markdown(
    """
    **Anggota Kelompok:**  
    📌 **10123038** - Muhamad Irsad Assopi  
    📌 **10123015** - Yoan Ready Syavera  
    📌 **10123028** - Rizky Al Farid Hafizh  
    📌 **10123020** - Hizkia Imanuel Edho  
    📌 **10123049** - Nur Ain Salimah  
    📌 **10123062** - Wa Ode Syahwa Salsabilah  
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown("🚀 **Universitas XYZ** | 2025")

# Konten Halaman
if menu == "Beranda":
    st.title("📦 Dashboard Analisis Data Pengiriman Pesanan")
    st.image("https://via.placeholder.com/800x300", use_column_width=True)
    st.markdown(
        """
        Selamat datang di Dashboard Analisis Data Pengiriman Pesanan.
        Dashboard ini menyajikan berbagai analisis terkait pengiriman pesanan.
        Silakan pilih menu di sebelah kiri untuk melihat detail analisis.
        """
    )

elif menu == "Analisis Data":
    st.title("📊 Analisis Data Pengiriman")
    
    @st.cache_data
    def load_data():
        url = 'https://raw.githubusercontent.com/Ims4d/sementara/refs/heads/main/orders_dataset.csv'
        df = pd.read_csv(url)
        df.fillna({
            'order_approved_at': pd.Timestamp.max, 
            'order_delivered_carrier_date': pd.Timestamp.max, 
            'order_delivered_customer_date': pd.Timestamp.max
        }, inplace=True)
        date_cols = ['order_purchase_timestamp', 'order_approved_at', 
                     'order_delivered_carrier_date', 'order_delivered_customer_date', 
                     'order_estimated_delivery_date']
        for col in date_cols:
            df[col] = pd.to_datetime(df[col])
        df['days_to_carrier'] = (df['order_delivered_carrier_date'] - df['order_purchase_timestamp']).dt.days
        return df
    
    df = load_data()
    
    st.subheader("📌 Informasi Dataset")
    col1, col2 = st.columns(2)
    col1.metric("📊 Total Data", f"{df.shape[0]} baris")
    col2.metric("📋 Total Kolom", f"{df.shape[1]}")
    
    option = st.selectbox("Pilih Pertanyaan Bisnis:", 
                          ["Rata-rata Waktu Pengiriman ke Kurir", 
                           "Persentase Pesanan Tepat Waktu vs Terlambat", 
                           "Distribusi Status Pengiriman"])
    
    if option == "Rata-rata Waktu Pengiriman ke Kurir":
        st.subheader("📦 Rata-rata Waktu Pengiriman ke Kurir")
        avg_time = df['days_to_carrier'].mean()
        st.markdown(f"### Rata-rata waktu pengiriman ke kurir: **{avg_time:.0f} hari**")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['days_to_carrier'].dropna(), bins=20, kde=True, color='blue')
        ax.set_title('Distribusi Waktu Pengiriman ke Kurir')
        ax.set_xlabel('Hari')
        ax.set_ylabel('Frekuensi')
        st.pyplot(fig)
    
elif menu == "Tentang Kelompok":
    st.title("📌 Tentang Kelompok")
    st.image("https://via.placeholder.com/800x400", use_column_width=True)
    st.markdown(
        """
        **Kelompok 2 - Analisis Data Pengiriman Pesanan**
        
        🔹 **Muhamad Irsad Assopi (10123038)**  
        🔹 **Yoan Ready Syavera (10123015)**  
        🔹 **Rizky Al Farid Hafizh (10123028)**  
        🔹 **Hizkia Imanuel Edho (10123020)**  
        🔹 **Nur Ain Salimah (10123049)**  
        🔹 **Wa Ode Syahwa Salsabilah (10123062)**  
        
        Proyek ini dibuat sebagai bagian dari analisis data untuk memahami pola pengiriman pesanan.
        """
    )
