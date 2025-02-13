import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import f_oneway
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

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
    def calculate_days_to_carrier(row):
        if row['order_status'] == 'delivered':
            return (row['order_delivered_carrier_date'] - row['order_purchase_timestamp']).days
        else:
            return np.nan
    df['days_to_carrier'] = df.apply(calculate_days_to_carrier, axis=1)
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
    st.title("Rata-rata Waktu Pengiriman ke Kurir")
    df = load_data()
    rata_rata_waktu_pengiriman = df['days_to_carrier'].mean()
    st.write(f"Rata-rata waktu pengiriman ke kurir: {rata_rata_waktu_pengiriman:.0f} hari")
    st.markdown("### Deskripsi Analisis")
    st.write("Analisis ini bertujuan untuk memahami seberapa lama waktu yang dibutuhkan sejak pembelian hingga pesanan dikirimkan ke kurir.")
    st.write("Dengan mengetahui rata-rata waktu ini, kita dapat mengidentifikasi potensi keterlambatan dalam sistem pengiriman.")
    st.markdown("### Prediksi Waktu Pengiriman Menggunakan Machine Learning")
    df.dropna(subset=['days_to_carrier'], inplace=True)
    df['order_purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
    df['order_purchase_hour'] = df['order_purchase_timestamp'].dt.hour
    
    X = df[['order_purchase_dayofweek', 'order_purchase_hour']]
    y = df['days_to_carrier']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    error = mean_absolute_error(y_test, y_pred)
    
    st.write(f"Model Prediksi Waktu Pengiriman menunjukkan error rata-rata sebesar {error:.2f} hari.")
    st.dataframe(df[['order_purchase_timestamp', 'order_delivered_carrier_date', 'days_to_carrier']].head())

def page_3():
    st.title("Estimasi Keterlambatan Pengiriman")
    st.text("Persentase pesanan yang dikirimkan terlambat ke pelanggan dibandingkan dengan estimasi waktu pengiriman")
    df = load_data()
    
    if 'delivery_duration' not in df.columns:
        df['delivery_duration'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

    if 'is_late' not in df.columns:
        df['is_late'] = (df['order_delivered_customer_date'] > df['order_estimated_delivery_date']).astype(int)

    try:
        df['month'] = df['order_delivered_customer_date'].dt.month
        late_orders_per_month = df[df['is_late'] == 1].groupby('month').size()
        fig_month = px.bar(x=late_orders_per_month.index, y=late_orders_per_month.values,
                         title='Jumlah Pesanan Terlambat per Bulan')
        st.plotly_chart(fig_month)
    except AttributeError:
        st.error("Tidak dapat menampilkan keterlambatan per bulan karena format data tidak sesuai.")

    try:
        df['day_of_week'] = df['order_delivered_customer_date'].dt.dayofweek
        late_orders_per_day = df[df['is_late'] == 1].groupby('day_of_week').size()
        fig_day = px.bar(x=late_orders_per_day.index, y=late_orders_per_day.values,
                         title='Jumlah Pesanan Terlambat per Hari dalam Seminggu')
        st.plotly_chart(fig_day)
    except AttributeError:
        st.error("Tidak dapat menampilkan keterlambatan per hari karena format data tidak sesuai.")

    df_cluster = df[['delivery_duration', 'is_late', 'month', 'day_of_week']].copy()
    df_cluster = df_cluster.dropna()
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_cluster)
    
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_cluster['cluster'] = kmeans.fit_predict(scaled_features)
    
    fig_3d = px.scatter_3d(df_cluster, x='delivery_duration', y='month', z='day_of_week', color='cluster',
                           title='Pengelompokan Pesanan Berdasarkan Durasi Pengiriman')
    st.plotly_chart(fig_3d)

    fig_box = px.box(df_cluster, x='cluster', y='delivery_duration',
                     title='Distribusi Durasi Pengiriman per Cluster')
    st.plotly_chart(fig_box)
    
    corr_matrix = df_cluster.corr()
    fig_heatmap = px.imshow(corr_matrix, labels=dict(x="Fitur", y="Fitur", color="Korelasi"),
                            x=corr_matrix.columns, y=corr_matrix.columns,
                            title='Heatmap Korelasi Antar Fitur')
    st.plotly_chart(fig_heatmap)

def page_4():
    st.title("Distribusi Status Pengiriman dan Clustering")
    df = load_data()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Distribusi Status Pengiriman")
        st.write("Grafik ini menunjukkan jumlah pesanan berdasarkan status pengiriman. Informasi ini berguna untuk memahami pola pemrosesan pesanan.")
        status_distribution = df['order_status'].value_counts()
        colors = {
            'approved': 'green', 'canceled': 'red', 'created': 'orange', 'delivered': 'green',
            'invoiced': 'orange', 'processing': 'orange', 'shipped': 'orange', 'unavailable': 'red'
        }
        fig, ax = plt.subplots(figsize=(6, 4))
        status_distribution.plot(kind='bar', color=[colors.get(status, 'gray') for status in status_distribution.index], ax=ax, edgecolor='black')
        ax.set_title('Distribusi Status Pengiriman')
        ax.set_xlabel('Status Pengiriman')
        ax.set_ylabel('Jumlah Pesanan')
        ax.set_xticklabels(status_distribution.index, rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Analisis Clustering")
        st.write("Clustering digunakan untuk memahami pola keterlambatan pengiriman berdasarkan waktu pembelian dan pengiriman pesanan.")
        
        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
        
        df['purchase_day'] = (df['order_purchase_timestamp'] - df['order_purchase_timestamp'].min()).dt.days
        df['delivery_day'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
        
        features = df[['purchase_day', 'delivery_day']].fillna(0)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        wcss = []
        K_range = range(1, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features_scaled)
            wcss.append(kmeans.inertia_)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(K_range, wcss, marker='o', linestyle='--')
        ax.set_xlabel('Jumlah Cluster (K)')
        ax.set_ylabel('WCSS')
        ax.set_title('Metode Elbow untuk Menentukan Jumlah Cluster Optimal')
        ax.set_xticks(K_range)
        st.pyplot(fig)
    
    st.markdown("### Hasil Clustering")
    st.write("Visualisasi hasil clustering dengan K=3 untuk mengelompokkan pola keterlambatan pengiriman.")
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(features_scaled)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(df['purchase_day'], df['delivery_day'], c=df['cluster'], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Cluster')
    ax.set_xlabel('Hari sejak awal transaksi')
    ax.set_ylabel('Lama pengiriman (hari)')
    ax.set_title('Clustering Pola Keterlambatan Pengiriman')
    st.pyplot(fig)

def page_5():
    st.title("Analisis Waktu Persetujuan Pesanan")
    df = load_data()
    
    st.write("## Deskripsi Dataset")
    st.write("Dataset ini menganalisis waktu yang dibutuhkan untuk menyetujui pesanan berdasarkan berbagai faktor.")
    
    def calculate_hour_to_approved(row):
        if row['order_status'] == 'delivered':
            return (row['order_approved_at'] - row['order_purchase_timestamp']).seconds / 3600
        else:
            return np.nan
    
    df['time_difference'] = df.apply(calculate_hour_to_approved, axis=1)
    
    # Buat fitur waktu tambahan
    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
    df['purchase_day'] = df['order_purchase_timestamp'].dt.day
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month
    df['hour_of_day'] = df['order_approved_at'].dt.hour
    df['day_of_week'] = df['order_approved_at'].dt.dayofweek
    df['month'] = df['order_approved_at'].dt.month
    df['days_to_carrier'] = (df['order_delivered_carrier_date'] - df['order_purchase_timestamp']).dt.days
    df['is_late'] = (df['order_delivered_carrier_date'] > df['order_estimated_delivery_date']).astype(int)
    
    # Fitur yang bisa mempengaruhi waktu persetujuan
    features = [
        'purchase_hour', 'purchase_day', 'purchase_month',
        'days_to_carrier', 'is_late',
        'hour_of_day', 'day_of_week', 'month'
    ]
    
    # Mengubah 'order_status' menjadi variabel numerik dengan One-Hot Encoding
    df = pd.get_dummies(df, columns=['order_status'], drop_first=True)
    features += [col for col in df.columns if 'order_status_' in col]
    df = df.dropna(subset=['time_difference'])
    X = df[features]
    y = df['time_difference']
    
    # Membagi data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluasi model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write("## Evaluasi Model")
    st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f} jam")
    st.metric(label="R² Score", value=f"{r2:.2f}")
    
    # Analisis Feature Importance
    feature_importance = model.feature_importances_
    feature_names = X.columns
    
    st.write("## Faktor yang Mempengaruhi Waktu Persetujuan")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=feature_names, palette="viridis", ax=ax)
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance Analysis")
    st.pyplot(fig)
    
    # Rata-rata waktu persetujuan
    rata_rata_waktu_disetujui = df['time_difference'].mean()
    st.write(f"Rata-rata waktu pesanan disetujui: {rata_rata_waktu_disetujui:.0f} jam")
    
    # Histogram selisih waktu persetujuan pesanan
    st.write("## Distribusi Waktu Persetujuan Pesanan")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['time_difference'].dropna(), bins=20, color='purple', log=True)
    ax.set_title("Histogram Waktu Persetujuan Pesanan (log scale)")
    ax.set_xlabel("Waktu Persetujuan (jam)")
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig)
    
    # Hitung rata-rata waktu persetujuan per jam
    avg_time_hour_of_day = df.groupby('hour_of_day')['time_difference'].mean()
    avg_time_purchase_hour = df.groupby('purchase_hour')['time_difference'].mean()
    
    # Visualisasi perbandingan
    st.write("## Perbandingan Waktu Persetujuan Berdasarkan Jam")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(avg_time_hour_of_day.index, avg_time_hour_of_day.values, marker='o', linestyle='-', label='Jam Persetujuan')
    ax.plot(avg_time_purchase_hour.index, avg_time_purchase_hour.values, marker='s', linestyle='--', label='Jam Pembelian')
    ax.set_xlabel("Jam dalam Sehari")
    ax.set_ylabel("Rata-rata Waktu Persetujuan (jam)")
    ax.set_title("Perbandingan Rata-rata Waktu Persetujuan")
    ax.set_xticks(range(0, 24))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    st.pyplot(fig)

def page_6():
    st.title("Pesanan Yang Berhasil Dikirim Dalam Satu Bulan Terakhir")
    df = load_data()
    last_month = df[df['order_delivered_customer_date'] >= pd.Timestamp.now() - pd.Timedelta('30 days')]
    total_delivered = last_month.shape[0]
    st.write(f"Jumlah pesanan yang berhasil dikirim dalam satu bulan terakhir: {total_delivered}")
    
    # Menyiapkan data untuk prediksi jumlah pesanan bulan berikutnya
    df['order_month'] = df['order_delivered_customer_date'].dt.to_period('M')
    orders_per_month = df.groupby('order_month').size().reset_index(name='order_count')
    
    # Konversi periode menjadi angka untuk model ML
    orders_per_month['order_month'] = orders_per_month['order_month'].astype(str)
    orders_per_month['order_month'] = pd.to_datetime(orders_per_month['order_month']).map(pd.Timestamp.toordinal)
    
    # Model regresi untuk prediksi jumlah pesanan bulan depan
    X = orders_per_month[['order_month']]
    y = orders_per_month['order_count']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Prediksi untuk bulan depan
    next_month = pd.Timestamp.now().to_period('M') + 1
    next_month_ordinal = pd.Timestamp(str(next_month)).toordinal()
    predicted_orders = model.predict([[next_month_ordinal]])[0]
    
    st.write(f"Prediksi jumlah pesanan yang akan dikirim bulan depan: {predicted_orders:.0f}")
    
    # Visualisasi tren pesanan per bulan
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(orders_per_month['order_month'], orders_per_month['order_count'], marker='o', linestyle='-', label='Data Aktual')
    ax.scatter(next_month_ordinal, predicted_orders, color='red', label='Prediksi', zorder=3)
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Jumlah Pesanan")
    ax.set_title("Tren Jumlah Pesanan Per Bulan")
    ax.legend()
    st.pyplot(fig)

def page_7():
    st.title("Analisis Waktu Pengiriman Berdasarkan Waktu Pembelian")
    
    # Simulasi Data (Bisa diganti dengan membaca file CSV)
    data = {
        "purchase_time": ["pagi", "siang", "malam", "pagi", "siang", "malam"],
        "delivery_time": [11, 8.59, 13.50, 5.12, 18.11, 22.27]  # Waktu pengiriman dalam hari
    }
    df = pd.DataFrame(data)
    
    # Statistik Deskriptif
    st.write("### Statistik Deskriptif")
    st.write(df.groupby("purchase_time")["delivery_time"].describe())
    
    # Visualisasi Distribusi Waktu Pengiriman
    st.write("### Distribusi Waktu Pengiriman Berdasarkan Waktu Pembelian")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="purchase_time", y="delivery_time", data=df, palette="Set2", ax=ax)
    ax.set_xlabel("Waktu Pembelian")
    ax.set_ylabel("Waktu Pengiriman (Hari)")
    ax.set_title("Distribusi Waktu Pengiriman Berdasarkan Waktu Pembelian")
    st.pyplot(fig)
    
    # Analisis Statistik: ANOVA
    purchase_groups = [group["delivery_time"].values for name, group in df.groupby("purchase_time")]
    anova_result = f_oneway(*purchase_groups)
    
    st.write("### Hasil Uji ANOVA")
    st.write(f"F-statistic: {anova_result.statistic}, p-value: {anova_result.pvalue}")
    
    # Interpretasi Hasil
    if anova_result.pvalue < 0.05:
        st.write("Ada perbedaan signifikan dalam waktu pengiriman berdasarkan waktu pembelian.")
    else:
        st.write("Tidak ada perbedaan signifikan dalam waktu pengiriman berdasarkan waktu pembelian.")


pg = st.navigation({
    "Navigasi Dashboard": [
        st.Page(page_1, title="Beranda"),
        st.Page(page_2, title="Rata-rata Waktu Pengiriman"),
        st.Page(page_3, title="Estimasi Keterlambatan Pengiriman"),
        st.Page(page_4, title="Distribusi Status Pengiriman"),
        st.Page(page_5, title="Rata-rata Waktu Pesanan Disetujui"),
        st.Page(page_6, title="Pesanan Dalam Satu Bulan Terakhir"),
        st.Page(page_6, title="Analisis Waktu Pengiriman Berdasarkan Waktu Pembelian")
    ]
})

pg.run()