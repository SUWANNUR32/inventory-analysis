import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# Konfigurasi Halaman
st.set_page_config(page_title="AI Inventory Optimizer", layout="wide")

st.title("📦 AI Smart Inventory & Demand Forecasting")
st.markdown("""
Aplikasi ini menggunakan **Machine Learning (Random Forest)** untuk menganalisis data penjualan historis 
dan memberikan rekomendasi stok barang secara otomatis.
""")

# --- SIDEBAR: INPUT DATA ---
st.sidebar.header("Input Data Penjualan")
with st.sidebar.form("input_form"):
    nama_barang = st.text_input("Nama Barang", placeholder="Contoh: Kaos Polos")
    ukuran = st.selectbox("Ukuran", ["S", "M", "L", "XL", "All Size"])
    jumlah_terjual = st.number_input("Jumlah Terjual", min_value=0, step=1)
    tanggal = st.date_input("Tanggal Penjualan")
    stok_skrg = st.number_input("Stok Saat Ini", min_value=0, step=1)
    submit_button = st.form_submit_button("Tambah ke Database")

# Inisialisasi State untuk menyimpan data
if 'data_history' not in st.session_state:
    st.session_state.data_history = pd.DataFrame(columns=[
        'tanggal', 'nama_barang', 'ukuran', 'jumlah_terjual', 'stok_saat_ini'
    ])

if submit_button:
    new_data = {
        'tanggal': pd.to_datetime(tanggal),
        'nama_barang': nama_barang,
        'ukuran': ukuran,
        'jumlah_terjual': jumlah_terjual,
        'stok_saat_ini': stok_skrg
    }
    st.session_state.data_history = pd.concat([st.session_state.data_history, pd.DataFrame([new_data])], ignore_index=True)
    st.success(f"Data {nama_barang} berhasil ditambahkan!")

# --- MAIN CONTENT: ANALISIS ---
if not st.session_state.data_history.empty:
    df = st.session_state.data_history
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Riwayat Penjualan")
        st.dataframe(df, use_container_width=True)

    # Proses AI
    results = []
    unique_items = df['nama_barang'].unique()

    for item in unique_items:
        df_item = df[df['nama_barang'] == item].copy()
        df_item['bulan'] = df_item['tanggal'].dt.month
        
        # AI Logic (Membutuhkan minimal 2 data poin untuk training sederhana)
        if len(df_item) >= 2:
            X = df_item[['bulan']]
            y = df_item['jumlah_terjual']
            
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X, y)
            
            next_month = (df_item['tanggal'].max().month % 12) + 1
            prediksi = max(0, round(model.predict([[next_month]])[0]))
            stok_ideal = round(prediksi * 1.25) # Safety stock 25%
            stok_skrg_item = df_item['stok_saat_ini'].iloc[-1]
            
            # Rekomendasi Logic
            if stok_skrg_item < stok_ideal:
                rekomendasi = "KULAKAN"
                urgensi = "Tinggi"
            elif stok_skrg_item > (stok_ideal * 1.5):
                rekomendasi = "KURANGI STOK"
                urgensi = "Rendah"
            else:
                rekomendasi = "AMAN"
                urgensi = "Normal"
                
            results.append({
                "Barang": item,
                "Prediksi Bulan Depan": prediksi,
                "Stok Ideal": stok_ideal,
                "Stok Saat Ini": stok_skrg_item,
                "Status": rekomendasi,
                "Urgensi": urgensi
            })

    if results:
        with col2:
            st.subheader("💡 Rekomendasi AI")
            res_df = pd.DataFrame(results)
            st.table(res_df)

        # Visualisasi Tren
        st.subheader("📈 Tren Penjualan per Barang")
        fig = px.line(df, x='tanggal', y='jumlah_terjual', color='nama_barang', markers=True)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Belum ada data. Silakan masukkan data penjualan di sidebar sebelah kiri.")
