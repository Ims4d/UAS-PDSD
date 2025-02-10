import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

st.set_page_config(
    page_title="Dashboard Analisis Data Pengiriman Pesanan",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul dan informasi kelompok
st.title("📦 Dashboard Analisis Data Pengiriman Pesanan")
st.markdown("## Kelompok 2 - Analisis Pengiriman")
st.markdown("---")

# Sidebar navigasi
st.sidebar.header("📌 Navigasi Dashboard")
option = st.sidebar.radio("Pilih Menu:", 
                          ["Dashboard", "Anggota Kelompok", "Pertanyaan 1", "Pertanyaan 2", "Pertanyaan 3", 
                           "Pertanyaan 4", "Pertanyaan 5", "Pertanyaan 6"])

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
    
    # Hitung selisih hari pengiriman ke kurir
    df['days_to_carrier'] = (df['order_delivered_carrier_date'] - df['order_purchase_timestamp']).dt.days
    
    return df

# Muat data
df = load_data()
if df.empty:
    st.error("❌ Data tidak dapat dimuat. Silakan periksa URL atau koneksi internet.")
    st.stop()

# Informasi dataset
st.sidebar.markdown("### ℹ️ Informasi Dataset")
st.sidebar.write(f"📊 Total Data: **{df.shape[0]}** baris")
st.sidebar.write(f"📋 Total Kolom: **{df.shape[1]}**")

if option == "Anggota Kelompok":
    st.title("📌 Anggota Kelompok 2")
    st.markdown("""
    **Kelompok 2 - Analisis Pengiriman**
    
    📌 **Anggota Kelompok:**
    - **10123038** - Muhamad Irsad Assopi  
    - **10123015** - Yoan Ready Syavera  
    - **10123028** - Rizky Al Farid Hafizh  
    - **10123020** - Hizkia Imanuel Edho  
    - **10123049** - Nur Ain Salimah  
    - **10123062** - Wa Ode Syahwa Salsabilah  
    
    🚀 **Universitas Komputer Indonesia** | 2025
    """)
    st.stop()

# Tampilan dashboard berdasarkan pilihan
elif option == "Pertanyaan 1":
    st.header("Pertanyaan 1: Rata-rata Waktu Pengiriman ke Kurir")
    
    # Hitung rata-rata waktu pengiriman
    avg_time = df['days_to_carrier'].mean()
    st.markdown(f"### Rata-rata waktu pengiriman ke kurir: **{avg_time:.0f} hari**")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['days_to_carrier'].dropna(), bins=20, color='skyblue', edgecolor='black')
    ax.set_title('Distribusi Waktu Pengiriman ke Kurir')
    ax.set_xlabel('Hari')
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)


elif option == "Pertanyaan 2":
    st.header("Pertanyaan 2: Persentase Pesanan Tepat Waktu vs Terlambat")
    
    # Identifikasi pesanan terlambat
    df['is_late'] = df['order_delivered_customer_date'] > df['order_estimated_delivery_date']
    late_percentage = (df['is_late'].sum() / len(df)) * 100
    on_time_percentage = 100 - late_percentage
    
    st.markdown(f"### Tepat Waktu: **{on_time_percentage:.1f}%** | Terlambat: **{late_percentage:.1f}%**")
    
    # Pie chart perbandingan
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['Tepat Waktu', 'Terlambat']
    sizes = [on_time_percentage, late_percentage]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0, 0.1)
    ax.pie(sizes, labels=labels, colors=colors, explode=explode,
           autopct='%1.1f%%', startangle=140)
    ax.set_title('Persentase Pesanan Tepat Waktu vs Terlambat')
    st.pyplot(fig)

elif option == "Pertanyaan 3":
    st.header("Pertanyaan 3: Distribusi Status Pengiriman")
    
    status_dist = df['order_status'].value_counts()
    st.markdown("### Distribusi Status Pengiriman")
    
    colors_dict = {
        'approved': '#27ae60', 'canceled': '#e74c3c', 'created': '#f39c12', 
        'delivered': '#27ae60', 'invoiced': '#f39c12', 'processing': '#f39c12', 
        'shipped': '#f39c12', 'unavailable': '#e74c3c'
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(status_dist.index, status_dist.values,
           color=[colors_dict.get(status, 'grey') for status in status_dist.index],
           edgecolor='black')
    ax.set_title('Distribusi Status Pengiriman')
    ax.set_xlabel('Status Pengiriman')
    ax.set_ylabel('Jumlah Pesanan')
    plt.setp(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

elif option == "Pertanyaan 4":
    st.header("Pertanyaan 4: Rata-rata Waktu Pesanan Disetujui")
    
    df['time_difference'] = (df['order_approved_at'] - df['order_purchase_timestamp']).dt.total_seconds() / 3600
    avg_diff = df['time_difference'].mean()
    st.markdown(f"### Rata-rata waktu persetujuan pesanan: **{avg_diff:.0f} jam**")
    
    # Histogram dengan skala logaritmik
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['time_difference'].dropna(), bins=20, color='mediumpurple', log=True)
    ax.set_title("Histogram Selisih Waktu Persetujuan (Jam) dengan Skala Log")
    ax.set_xlabel("Selisih Waktu (jam)")
    ax.set_ylabel("Frekuensi (log scale)")
    st.pyplot(fig)

elif option == "Pertanyaan 5":
    st.header("Pertanyaan 5: Jumlah Pesanan yang Berhasil Dikirim dalam Satu Bulan Terakhir")
    
    # Filter pesanan yang dikirim dalam 30 hari terakhir
    last_month = df[df['order_delivered_customer_date'] >= pd.Timestamp.now() - pd.Timedelta(days=30)]
    total_delivered = last_month.shape[0]
    st.markdown(f"### Jumlah pesanan yang berhasil dikirim dalam satu bulan terakhir: **{total_delivered}**")


elif option == "Pertanyaan 6":
    st.header("Pertanyaan 6: Perbedaan Waktu Pengiriman Berdasarkan Waktu Pembelian")
    
    # Klasifikasikan waktu pembelian berdasarkan jam
    df['purchase_time'] = df['order_purchase_timestamp'].dt.hour.apply(
        lambda x: 'Pagi' if 6 <= x < 12 else 'Siang' if 12 <= x < 18 else 'Malam'
    )
    
    # Boxplot distribusi waktu pengiriman per kategori waktu pembelian
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="purchase_time", y="days_to_carrier", data=df, palette="Set2", ax=ax)
    ax.set_title("Distribusi Waktu Pengiriman Berdasarkan Waktu Pembelian")
    ax.set_xlabel("Waktu Pembelian")
    ax.set_ylabel("Waktu Pengiriman (Hari)")
    st.pyplot(fig)
    
    # Uji ANOVA untuk mengetahui perbedaan signifikan
    purchase_groups = [group["days_to_carrier"].dropna().values 
                       for _, group in df.groupby("purchase_time")]
    anova_result = f_oneway(*purchase_groups)
    
    st.markdown("#### Hasil Uji ANOVA")
    st.write(f"F-statistic: **{anova_result.statistic:.4f}**")
    st.write(f"p-value: **{anova_result.pvalue:.4f}**")
    
    if anova_result.pvalue < 0.05:
        st.success("Ada perbedaan signifikan dalam waktu pengiriman berdasarkan waktu pembelian.")
    else:
        st.info("Tidak ada perbedaan signifikan dalam waktu pengiriman berdasarkan waktu pembelian.")

st.sidebar.markdown("---")
st.sidebar.markdown("📌 **Dibuat oleh Kelompok 2**")
st.sidebar.markdown("🚀 **Universitas Komputer Indonesia** | 2025")
