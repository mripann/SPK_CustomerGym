import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gym Member Clustering", layout="wide")

st.title("SPK: Clustering Anggota Gym Menggunakan K-Means")
st.markdown("""
Aplikasi ini melakukan **Clustering (K-Means)** untuk mengelompokkan anggota gym 
berdasarkan pola latihan, demografi, atau fitur lain pada dataset.
""")

@st.cache_data
def load_data_from_file(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data
def load_data_from_path(path):
    return pd.read_csv(path)

# Load dataset
uploaded_file = st.sidebar.file_uploader("Unggah dataset CSV", type=["csv"])
if uploaded_file is not None:
    df = load_data_from_file(uploaded_file)
    st.sidebar.success("Dataset berhasil diunggah.")
else:
    default_path = "data/gym_members_exercise_tracking.csv"
    try:
        df = load_data_from_path(default_path)
        st.sidebar.info(f"Menggunakan dataset default: {default_path}")
    except:
        st.error("Silakan unggah dataset terlebih dahulu.")
        st.stop()

st.subheader("Preview Dataset")
st.dataframe(df.head())

# --- Preprocessing ---
st.subheader("Preprocessing Data")

numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()

# pilih fitur
selected_features = st.multiselect(
    "Pilih fitur untuk clustering",
    numeric_features + non_numeric,
    default=numeric_features
)

# salin data
df_proc = df.copy()

# menangani missing values
for col in selected_features:
    if df_proc[col].isna().sum() > 0:
        if df_proc[col].dtype != object:
            df_proc[col] = df_proc[col].fillna(df_proc[col].mean())
        else:
            df_proc[col] = df_proc[col].fillna(df_proc[col].mode()[0])

# Encode categorical features
label_encoders = {}
for col in selected_features:
    if df_proc[col].dtype == object:
        le = LabelEncoder()
        df_proc[col] = le.fit_transform(df_proc[col])
        label_encoders[col] = le

st.write("Preview setelah preprocessing:")
st.dataframe(df_proc[selected_features].head())

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_proc[selected_features])

# --- Clustering ---
st.subheader("Clustering - KMeans")

k = st.slider("Pilih jumlah cluster (K)", 2, 10, 3)
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

df_proc["cluster"] = cluster_labels

st.write("Distribusi Cluster:")
st.write(df_proc["cluster"].value_counts())

# Ringkasan cluster
st.subheader("Ringkasan tiap cluster")
cluster_summary = df_proc.groupby("cluster")[selected_features].mean().round(3)
st.dataframe(cluster_summary)

# Visualization PCA 2D
st.subheader("Visualisasi Cluster (PCA 2D)")

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df_proc["pca1"] = pca_result[:, 0]
df_proc["pca2"] = pca_result[:, 1]

fig, ax = plt.subplots(figsize=(8, 5))
for c in df_proc["cluster"].unique():
    cluster_points = df_proc[df_proc["cluster"] == c]
    ax.scatter(cluster_points["pca1"], cluster_points["pca2"], label=f"Cluster {c}", alpha=0.7)

ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.legend()
st.pyplot(fig)

# download clustering result
st.download_button(
    label="Download Dataset dengan Cluster",
    data=df_proc.to_csv(index=False).encode("utf-8"),
    file_name="gym_members_clustered.csv",
    mime="text/csv"
)

st.caption("Clustering-only.")
