import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Dashboard Analisis Data Pengiriman Pesanan", layout="wide")

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

df = load_data()

def page_1():
    st.title("Beranda")
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
    st.title("Distribusi Status Pengiriman")
    status_distribution = df['order_status'].value_counts()
    colors = {
        'approved': 'green', 'canceled': 'red', 'created': 'orange', 'delivered': 'green',
        'invoiced': 'orange', 'processing': 'orange', 'shipped': 'orange', 'unavailable': 'red'
    }
    fig, ax = plt.subplots()
    status_distribution.plot(kind='bar', color=[colors.get(status, 'gray') for status in status_distribution.index], ax=ax, edgecolor='black')
    ax.set_title('Distribusi Status Pengiriman')
    ax.set_xlabel('Status Pengiriman')
    ax.set_ylabel('Jumlah Pesanan')
    ax.set_xticklabels(status_distribution.index, rotation=45)
    st.pyplot(fig)

def page_4():
    st.title("Rata-rata Waktu Persetujuan Pesanan")
    df['time_difference'] = (df['order_approved_at'] - df['order_purchase_timestamp']).dt.total_seconds() / 3600
    rata_rata_waktu_disetujui = df['time_difference'].mean()
    st.write(f"Rata-rata waktu pesanan disetujui: {rata_rata_waktu_disetujui:.0f} jam")
    fig, ax = plt.subplots()
    ax.hist(df['time_difference'].dropna(), bins=20, color='purple', log=True)
    ax.set_title("Histogram Selisih Waktu Persetujuan Pesanan (dalam Jam) dengan Skala Logaritmik")
    ax.set_xlabel("Selisih Waktu (jam)")
    ax.set_ylabel("Frekuensi (log scale)")
    st.pyplot(fig)

def page_5():
    st.title("Faktor yang Mempengaruhi Waktu Persetujuan Pesanan")
    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
    df['purchase_day'] = df['order_purchase_timestamp'].dt.day
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month
    df['hour_of_day'] = df['order_approved_at'].dt.hour
    df['day_of_week'] = df['order_approved_at'].dt.dayofweek
    df['month'] = df['order_approved_at'].dt.month
    df['days_to_carrier'] = (df['order_delivered_carrier_date'] - df['order_purchase_timestamp']).dt.days
    df['is_late'] = (df['order_delivered_carrier_date'] > df['order_estimated_delivery_date']).astype(int)
    features = ['purchase_hour', 'purchase_day', 'purchase_month', 'days_to_carrier', 'is_late', 'hour_of_day', 'day_of_week', 'month']
    df = pd.get_dummies(df, columns=['order_status'], drop_first=True)
    features += [col for col in df.columns if 'order_status_' in col]
    df = df.dropna(subset=['time_difference'])
    X = df[features]
    y = df['time_difference']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f'Mean Absolute Error: {mae:.2f} jam')
    st.write(f'R² Score: {r2:.2f}')
    feature_importance = model.feature_importances_
    feature_names = X.columns
    fig, ax = plt.subplots()
    sns.barplot(x=feature_importance, y=feature_names, palette="viridis", ax=ax)
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Faktor yang Mempengaruhi Waktu Persetujuan Pesanan")
    st.pyplot(fig)

def page_6():
    st.title("Informasi Kelompok")
    st.write("Data tentang anggota kelompok akan ditampilkan di sini.")

pages = {
    "Beranda": page_1,
    "Rata-rata Waktu Pengiriman": page_2,
    "Distribusi Status Pengiriman": page_3,
    "Rata-rata Waktu Persetujuan": page_4,
    "Faktor Waktu Persetujuan": page_5,
    "Informasi Kelompok": page_6
}

page = st.sidebar.radio("Navigasi Dashboard", list(pages.keys()))
pages[page]()
