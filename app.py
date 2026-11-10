import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn

MODEL_PATH = os.path.join("model", "customer_churn_model.pth")
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "InternetService",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

st.set_page_config(page_title="Prediksi Pelanggan", page_icon="📊", layout="wide")

st.markdown("""
<style>
.stApp { background: #f5f7fb; }
.block-container { padding-top: 2rem; max-width: 1180px; }
.hero { background: #101828; color: white; padding: 28px 32px; border-radius: 18px; margin-bottom: 22px; }
.hero h1 { margin: 0; font-size: 2.1rem; }
.hero p { color: #cbd5e1; margin: 8px 0 0; }
.result-risk { background: #fff1f2; border: 1px solid #fecdd3; padding: 22px; border-radius: 14px; }
.result-safe { background: #ecfdf5; border: 1px solid #a7f3d0; padding: 22px; border-radius: 14px; }
div.stButton > button { background: #2563eb; color: white; border: 0; border-radius: 10px; padding: .7rem 1.2rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>Tugas Besar</h1>
<p>M Fadlullah Assalafy Y • 202343501936</p>
</div>
""", unsafe_allow_html=True)

st.title("Prediksi Pelanggan Berhenti Berlangganan")
st.caption("Masukkan informasi pelanggan untuk mengetahui kemungkinan pelanggan berhenti menggunakan layanan.")

class ChurnNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)

def encode_input(frame, metadata):
    x = frame.copy()
    for col in NUMERIC:
        x[col] = pd.to_numeric(x[col], errors="coerce")
        x[col] = x[col].fillna(metadata["numeric_medians"][col])

    for col in CATEGORICAL:
        x[col] = x[col].fillna(metadata["category_modes"][col]).astype(str)

    numeric_array = np.column_stack([
        (x[col].astype(float).to_numpy() - metadata["numeric_means"][col])
        / metadata["numeric_stds"][col]
        for col in NUMERIC
    ]).astype(np.float32)

    categorical_arrays = []
    for col in CATEGORICAL:
        categories = metadata["category_values"][col]
        lookup = {value: idx for idx, value in enumerate(categories)}
        one_hot = np.zeros((len(x), len(categories)), dtype=np.float32)
        for row_idx, value in enumerate(x[col].astype(str).to_numpy()):
            if value in lookup:
                one_hot[row_idx, lookup[value]] = 1.0
        categorical_arrays.append(one_hot)

    return np.concatenate([numeric_array] + categorical_arrays, axis=1)

@st.cache_resource
def load_model():
    # weights_only=True menjaga pemuatan checkpoint berbasis tensor dan tipe data sederhana.
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = ChurnNeuralNetwork(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint

if not os.path.exists(MODEL_PATH):
    st.error("File model belum ditemukan. Letakkan customer_churn_model.pth di dalam folder model.")
    st.code("python3 train_model.py\nstreamlit run app.py", language="bash")
    st.stop()

model, checkpoint = load_model()

with st.form("form_prediksi"):
    st.subheader("Informasi Pelanggan")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender_label = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        senior_label = st.selectbox("Termasuk Lansia", ["Tidak", "Ya"])
        partner_label = st.selectbox("Memiliki Pasangan", ["Ya", "Tidak"])
        dependents_label = st.selectbox("Memiliki Tanggungan", ["Ya", "Tidak"])

    with col2:
        tenure = st.number_input("Lama Berlangganan (Bulan)", 0, 100, 24)
        phone_label = st.selectbox("Menggunakan Layanan Telepon", ["Ya", "Tidak"])
        internet_label = st.selectbox("Jenis Layanan Internet", ["DSL", "Fiber Optic", "Tidak Menggunakan"])
        contract_label = st.selectbox("Jenis Kontrak", ["Bulanan", "Satu Tahun", "Dua Tahun"])

    with col3:
        paperless_label = st.selectbox("Tagihan Tanpa Kertas", ["Ya", "Tidak"])
        payment_label = st.selectbox("Metode Pembayaran", [
            "Cek Elektronik", "Cek Melalui Pos",
            "Transfer Bank Otomatis", "Kartu Kredit Otomatis"
        ])
        monthly = st.number_input("Biaya Bulanan", min_value=0.0, value=70.50, step=0.5)
        total = st.number_input("Total Biaya Berlangganan", min_value=0.0, value=1692.0, step=1.0)

    submitted = st.form_submit_button("PREDIKSI SEKARANG", use_container_width=True)

if submitted:
    input_data = pd.DataFrame([{
        "gender": "Male" if gender_label == "Laki-laki" else "Female",
        "SeniorCitizen": 1 if senior_label == "Ya" else 0,
        "Partner": "Yes" if partner_label == "Ya" else "No",
        "Dependents": "Yes" if dependents_label == "Ya" else "No",
        "tenure": tenure,
        "PhoneService": "Yes" if phone_label == "Ya" else "No",
        "InternetService": {
            "DSL": "DSL", "Fiber Optic": "Fiber optic",
            "Tidak Menggunakan": "No"
        }[internet_label],
        "Contract": {
            "Bulanan": "Month-to-month", "Satu Tahun": "One year",
            "Dua Tahun": "Two year"
        }[contract_label],
        "PaperlessBilling": "Yes" if paperless_label == "Ya" else "No",
        "PaymentMethod": {
            "Cek Elektronik": "Electronic check",
            "Cek Melalui Pos": "Mailed check",
            "Transfer Bank Otomatis": "Bank transfer (automatic)",
            "Kartu Kredit Otomatis": "Credit card (automatic)"
        }[payment_label],
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])

    features = encode_input(input_data, checkpoint["metadata"])
    tensor = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor)).item()

    prediction = int(probability >= checkpoint.get("threshold", 0.5))
    st.divider()
    st.subheader("Hasil Prediksi")

    if prediction == 1:
        st.markdown("""
        <div class="result-risk">
        <h3 style="color:#be123c;margin-top:0">⚠ Berpotensi Berhenti Berlangganan</h3>
        <p>Pelanggan diprediksi memiliki kemungkinan untuk berhenti menggunakan layanan.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-safe">
        <h3 style="color:#047857;margin-top:0">✓ Tidak Berpotensi Berhenti Berlangganan</h3>
        <p>Pelanggan diprediksi memiliki kemungkinan untuk tetap menggunakan layanan.</p>
        </div>
        """, unsafe_allow_html=True)

    st.metric("Kemungkinan Berhenti Berlangganan", f"{probability * 100:.2f}%")
    st.progress(float(np.clip(probability, 0.0, 1.0)))
    st.info("Hasil merupakan prediksi model machine learning, bukan kepastian perilaku pelanggan.")

with st.expander("Tentang Aplikasi"):
    st.write("""
    Aplikasi ini menggunakan jaringan saraf tiruan (Artificial Neural Network)
    yang dibangun dengan PyTorch. Model dilatih menggunakan data Telco Customer
    Churn dan disimpan dalam format .pth untuk digunakan kembali saat prediksi.
    """)
