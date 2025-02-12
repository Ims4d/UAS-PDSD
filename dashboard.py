import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Dashboard Analisis Data Pengiriman Pesanan",
    layout="wide"
)

@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/Ims4d/sementara/refs/heads/main/orders_dataset.csv'
    df = pd.read_csv(url)
    df.fillna({
        'order_approved_at': pd.Timestamp.max, 
        'order_delivered_carrier_date': pd.Timestamp.max, 
        'order_delivered_customer_date': pd.Timestamp.max
    }, inplace=True)
    date_cols = [
        'order_purchase_timestamp',
        'order_approved_at', 
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    df['days_to_carrier'] = (df['order_delivered_carrier_date'] - df['order_purchase_timestamp']).dt.days
    return df

def page_1():
    st.title("Beranda")
    df = load_data()
    df['order_purchase_day'] = df['order_purchase_timestamp'].dt.date
    orders_per_day = df.groupby('order_purchase_day').size()
    fig, ax = plt.subplots(figsize=(12, 6))
    orders_per_day.plot(ax=ax)
    ax.set_title('Order Per Hari')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Banyak Order')
    st.pyplot(fig)

def page_2():
    st.title("Rata-rata Waktu Pengiriman Pesanan")
    df = load_data()
    df['is_late'] = df['order_delivered_customer_date'] > df['order_estimated_delivery_date']
    late_percentage = (df['is_late'].sum() / len(df)) * 100
    on_time_percentage = 100 - late_percentage
    labels = ['Tepat Waktu', 'Terlambat']
    sizes = [on_time_percentage, late_percentage]
    colors = ['green', 'red']
    explode = (0, 0.1)
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', startangle=140)
    ax.set_title('Persentase Pesanan Tepat Waktu vs Terlambat')
    st.pyplot(fig)

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

pg = st.navigation({
    "Navigasi Dashboard": [
        st.Page(page_1, title="Beranda"),
        st.Page(page_2, title="Rata-rata Waktu Pengiriman"),
        st.Page(page_3, title="Estimasi Keterlambatan Pengiriman"),
        st.Page(page_4, title="Distribusi Status Pengiriman"),
        st.Page(page_5, title="Rata-rata Waktu Pesanan Dibuat-Disetujui"),
        st.Page(page_6, title="Pesanan Dalam Satu Bulan Terakhir"),
        st.Page(page_7, title="Perbedaan Waktu Pengiriman Berdasarkan Waktu Pembelian")
    ],
    "Informasi Kelompok": [ st.Page(page_8, title="Anggota Kelompok") ]
})

pg.run()
