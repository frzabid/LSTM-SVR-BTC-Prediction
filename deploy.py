import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from new_svr_fiks import hasil_prediksi
from vmdxlstm import Hasil_prediksi_lstm_plot

# Fungsi untuk mengambil berita dari API
def fetch_news(query="Bitcoin"): 
    API_KEY = "d54f31ab690946cd8f737d93a9005184"  
    URL = "https://newsapi.org/v2/everything"
    
    params = {
        "q": query,
        "apiKey": API_KEY,
        "sortBy": "publishedAt",
        "language": "en",
    }
    response = requests.get(URL, params=params)
    
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        filtered_articles = [
            article for article in articles
            if article.get("urlToImage")
        ]
        unique_articles = {article['title']: article for article in filtered_articles}.values()
        return list(unique_articles)
    else:
        st.error(f"Error {response.status_code}: {response.text}")
        return []

# Set page configuration
st.set_page_config(page_title="Analisis Prediksi Harga Bitcoin", layout="wide")

# Custom CSS
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    font-size: 18px;
    font-weight: bold;
    color: #333;
    background-color: transparent;
    padding: 10px 20px;
    border: none;
    border-radius: 0;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #1E90FF;
    border-bottom: 3px solid #1E90FF;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #1E90FF;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("Analisis Prediksi Harga Bitcoin")

# Model selection
models = ["LSTM", "Linear Support Vector Regression"]
selected_model = st.selectbox("Pilih Model", models)

# Model metrics
model_metrics = {
    "LSTM": {
        "RMSE": {"value": 4349.804, "delta": -70.1},
        "MAE": {"value": 3160.402, "delta": -70.5},
        "R² Score": {"value": 0.9217, "delta": -7.0},
        "MAPE": {"value": 3.71, "delta": -82.6}
    },
    "Linear Support Vector Regression": {
        "RMSE": {"value": 1455.563, "delta": 70.1},
        "MAE": {"value": 1070.651, "delta": 70.5},
        "R² Score": {"value": 0.991, "delta": 7.0},
        "MAPE": {"value": 21.39, "delta": 82.6}
    }
}


# Detail model
if selected_model == "LSTM":
    st.header("Detail Model LSTM")
    st.subheader("Parameter Model")
    st.write("""
    - **Fitur Input**: 
        - BTC_High_BTC,BTC_Low_BTC,BTC_Open_BTC,BTC_Volume_BTC,
        - GSPC_Close_GSPC,GSPC_High_GSPC,GSPC_Low_GSPC,GSPC_Open_GSPC,GSPC_Volume_GSPC,
        - VIX_Close_VIX,VIX_High_VIX,VIX_Low_VIX,VIX_Open_VIX,
        - XAU_Close_XAU,XAU_High_XAU,XAU_Low_XAU,XAU_Open_XAU,XAU_Volume_XAU,value,
        - Mode_1,Mode_2,Mode_3,Mode_4,Mode_5,Mode_6,Mode_7,Mode_8
    - **Target**: 'BTC_Close_BTC'
    - **Normalisasi**: StandardScaler
    - **Train-Test Split**: 80% : 20%
    - **Regularisasi**: L2 (Ridge), alpha=0.01
    """)
else:
    st.header("Detail Model Linear Support Vector Regression")
    st.subheader("Parameter Model")
    st.write("""
    - **Fitur Input**: 
        - 'BTC_High_BTC', 'BTC_Low_BTC', 'BTC_Open_BTC' 
        - 'GSPC_High_GSPC', 'GSPC_Close_GSPC', 'GSPC_Open_GSPC' 
        - 'BTC_BB_Lower_BTC', 'GSPC_High_GSPC', 'GSPC_Close_GSPC', 'GSPC_Open_GSPC' 
        - 'GSPC_Low_GSPC', 'XAU_Low_XAU', 'XAU_Open_XAU', 'XAU_Close_XAU'
        - 'XAU_High_XAU', 'XAU_Volume_XAU'
    - **Target**: BTC_Close_BTC
    - **Normalisasi**: StandarScaler
    - **Train-Test Split**: 80% : 20%
    """)

# Koefisien fitur
st.subheader("Koefisien Fitur")
features = ["BTC_High_BTC", "BTC_Low_BTC", "BTC_Open_BTC", "BTC_EMA20_BTC", "BTC_MA20_BTC", 
            "BTC_BB_Middle_BTC", "BTC_BB_Upper_BTC", "BTC_SAR_BTC", "BTC_BB_Lower_BTC", 
            "GSPC_High_GSPC", "GSPC_Close_GSPC", "GSPC_Open_GSPC", "GSPC_Low_GSPC", 
            "XAU_Low_XAU", "XAU_Open_XAU", "XAU_Close_XAU", "XAU_High_XAU", "XAU_Volume_XAU"]

coefficients = [0.999248, 0.999060, 0.998228, 0.992420, 0.989816, 
                0.989816, 0.988883, 0.980306, 0.979922, 0.904512, 
                0.903351, 0.903147, 0.901735, 0.790028, 0.788821, 
                0.788162, 0.786683, 0.344684]

fig = go.Figure(go.Bar(
    x=features,
    y=coefficients,
    marker=dict(color='lightseagreen', opacity=0.7)
))
fig.update_layout(
    title="Koefisien Fitur",
    xaxis_title="Fitur",
    yaxis_title="Koefisien",
    template="plotly_dark",
    xaxis_tickangle=45
)
st.plotly_chart(fig)

# Header
st.header("Metrik Performa")

# Layout kolom
col1, col2, col3, col4 = st.columns(4)

# Ambil metrik berdasarkan model yang dipilih
metrics = model_metrics[selected_model]

# Tampilkan masing-masing metrik
with col1:
    st.metric("RMSE", f"{metrics['RMSE']['value']:.3f} USD", f"{metrics['RMSE']['delta']:.1f}%")
with col2:
    st.metric("MAE", f"{metrics['MAE']['value']:.3f} USD", f"{metrics['MAE']['delta']:.1f}%")
with col3:
    st.metric("R² Score", f"{metrics['R² Score']['value']:.4f}", f"{metrics['R² Score']['delta']:.1f}%")
with col4:
    st.metric("MAPE", f"{metrics['MAPE']['value']:.2f}%", f"{metrics['MAPE']['delta']:.1f}%")


# Grafik harga close
st.header("Grafik Harga Close Bitcoin")
df = pd.read_csv("df_merged_cleaned.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['BTC_Close_BTC'], mode='lines', name='Aktual', line=dict(color='blue')))
fig.update_layout(
    title='Data Close BTC',
    xaxis_title='Tanggal',
    yaxis_title='BTC_Close_BTC (USD)',
    xaxis_tickangle=45,
    template='plotly_dark'
)
st.plotly_chart(fig)

# Perbandingan prediksi
st.header("Perbandingan Harga Bitcoin Aktual vs Prediksi")
if selected_model == "LSTM":
    hasil_predik_lstm = Hasil_prediksi_lstm_plot('merged_lstm_vmd.csv')
    st.plotly_chart(hasil_predik_lstm)
else:
    hasil_predik_svr = hasil_prediksi('df_merged_cleaned.csv')
    st.plotly_chart(hasil_predik_svr)

st.header("Berita Terkini tentang Bitcoin")
articles = fetch_news("Bitcoin")

if articles:
    for article in articles[:5]:
        st.subheader(article["title"])

        cols = st.columns([6, 12])  # Gambar lebih kecil, teks lebih lebar

        with cols[0]:
            if article["urlToImage"]:
                st.image(article["urlToImage"], width=300)

        with cols[1]:
            st.write(article["description"])
            st.markdown(f"[Baca selengkapnya]({article['url']})")

        st.markdown("---")
else:
    st.write("Tidak ada berita yang ditemukan.")


# Footer
st.markdown("---")
# st.write("Analisis Prediksi Bitcoin Menggunakan LSTM dan SVR. Dibuat untuk Tugas Akhir Machine Learning.")
