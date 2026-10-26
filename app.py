"""
EE136IU Capstone Mini-Project - Topic 1: Comprehensive Model Benchmark
Automatic Modulation Classification (AMC) Across 8 Digital Modulation Schemes
Unified Interactive Dashboard:
  - Tier 1: Classical ML (Random Forest on 18 Domain Features)
  - Tier 2: Neural Network (3-Layer MLP on 18 Domain Features)
  - Tier 3: Deep Learning (1D CNN on Raw Baseband I/Q Tensors [2, 128])
Now Integrated with Full 96,000-Burst Dataset (Train, Val, Test Splits)!
"""

import os
import sys
import time
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

# -------------------------------------------------------------
# 1. Page Configuration & Professional Academic CSS
# -------------------------------------------------------------
st.set_page_config(
    page_title="EE136IU • Topic 1: Comprehensive Model Benchmark",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Academic Design System (Clean, Minimalist, Publication-Grade)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .code-font {
        font-family: 'JetBrains Mono', monospace;
    }

    .main-header {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 24px 28px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    
    .main-header h1 {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #F8FAFC;
    }
    
    .main-header p {
        font-size: 0.98rem;
        color: #94A3B8;
        margin-top: 6px;
        margin-bottom: 0;
        font-weight: 400;
    }

    .tag-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-right: 6px;
        margin-top: 12px;
        letter-spacing: 0.04em;
    }
    
    .tag-blue { background: #1E293B; color: #38BDF8; border: 1px solid #334155; }
    .tag-green { background: #1E293B; color: #34D399; border: 1px solid #334155; }
    .tag-purple { background: #1E293B; color: #A78BFA; border: 1px solid #334155; }
    .tag-pink { background: #1E293B; color: #F472B6; border: 1px solid #334155; }
    .tag-amber { background: #1E293B; color: #FBBF24; border: 1px solid #334155; }

    .model-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        height: 100%;
    }
    
    .badge-correct {
        background-color: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        letter-spacing: 0.03em;
        display: inline-block;
    }
    
    .badge-wrong {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        letter-spacing: 0.03em;
        display: inline-block;
    }
    
    .stat-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .ground-truth-banner {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #0284C7;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 20px;
    }

    .dataset-metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Directory Resolution & Model Architectures Definition
# -------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(ROOT_DIR, "ML")
MLP_DIR = os.path.join(ROOT_DIR, "MLP")
if not os.path.exists(MLP_DIR):
    MLP_DIR = os.path.join(ROOT_DIR, "emerging_project-main", "emerging_project-main")
CNN_DIR = os.path.join(ROOT_DIR, "EE136_1D_CNN")
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
FEAT_DIR = os.path.join(ROOT_DIR, "features_train")

# Tier 2: 3-Layer MLP Architecture
class ModulationMLP(nn.Module):
    def __init__(self, input_dim: int = 18, num_classes: int = 8):
        super(ModulationMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# Tier 3: 1D CNN Architecture (Raw I/Q Baseband [2, 128])
class CNN1D(nn.Module):
    def __init__(self, num_classes: int = 8):
        super(CNN1D, self).__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=5, padding=2),
                nn.BatchNorm1d(cout),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
        self.features = nn.Sequential(
            block(2, 32),
            block(32, 64),
            block(64, 128)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

# -------------------------------------------------------------
# 3. Model Loading & Data Caching
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Đang nạp 3 mô hình (Random Forest, 3-Layer MLP, 1D CNN)...")
def load_all_models():
    """Tải đồng thời Random Forest, 3-Layer MLP và 1D CNN"""
    # 1. Load Tier 1: Random Forest
    rf_path = os.path.join(ML_DIR, "rf_model.pkl")
    scaler_path = os.path.join(ML_DIR, "scaler.pkl")
    le_path = os.path.join(ML_DIR, "label_encoder.pkl")
    
    rf_model = joblib.load(rf_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(le_path)
    classes = list(label_encoder.classes_)
    
    # 2. Load Tier 2: 3-Layer MLP
    with open(os.path.join(MLP_DIR, "class_mapping.pkl"), "rb") as f:
        mapping = pickle.load(f)
    feature_cols = mapping["features"]
    
    mlp_model = ModulationMLP(len(feature_cols), len(classes))
    mlp_weights = torch.load(os.path.join(MLP_DIR, "best_mlp.pth"), map_location="cpu", weights_only=False)
    mlp_model.load_state_dict(mlp_weights)
    mlp_model.eval()

    # 3. Load Tier 3: 1D CNN
    cnn_model = CNN1D(len(classes))
    cnn_weights = torch.load(os.path.join(CNN_DIR, "cnn1d_best.pt"), map_location="cpu", weights_only=False)
    cnn_model.load_state_dict(cnn_weights["model_state_dict"])
    cnn_model.eval()
    
    return rf_model, mlp_model, cnn_model, scaler, label_encoder, classes, feature_cols

@st.cache_data(show_spinner="Đang nạp tập dữ liệu kiểm thử (14,400 bursts test)...")
def load_test_dataset():
    """Ưu tiên đọc tập test đầy đủ 14,400 mẫu từ folder features_train"""
    full_csv = os.path.join(FEAT_DIR, "features_test.csv")
    full_iq = os.path.join(FEAT_DIR, "iq_test.npz")
    
    if os.path.exists(full_csv) and os.path.exists(full_iq):
        df = pd.read_csv(full_csv)
        # Sử dụng mmap_mode để nạp cực nhanh với chi phí bộ nhớ tối thiểu
        npz = np.load(full_iq, mmap_mode="r")
        iq_array = npz["iq"]
        is_full = True
    else:
        # Fallback về tập con 600 bursts
        sample_csv = os.path.join(MLP_DIR, "sample_test_bursts.csv")
        df = pd.read_csv(sample_csv)
        iq_path = os.path.join(ASSETS_DIR, "sample_test_iq.npz")
        if os.path.exists(iq_path):
            iq_array = np.load(iq_path)["iq"]
        else:
            iq_array = np.zeros((len(df), 2, 128), dtype=np.float32)
        is_full = False
        
    return df, iq_array, is_full

@st.cache_data
def load_benchmark_metrics():
    comp_path = os.path.join(ASSETS_DIR, "models_benchmark_comparison_3tier.csv")
    if os.path.exists(comp_path):
        return pd.read_csv(comp_path)
    return None

rf_model, mlp_model, cnn_model, scaler, label_encoder, classes, feature_cols = load_all_models()
df_test, iq_bursts, is_full_data = load_test_dataset()
df_bench = load_benchmark_metrics()

# -------------------------------------------------------------
# 4. Header Banner
# -------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>EE136IU • Topic 1: Comprehensive Model Benchmark</h1>
    <p>Automatic Modulation Classification: Classical ML vs. MLP vs. 1D CNN Across 8 Digital Modulation Schemes</p>
    <div>
        <span class="tag-badge tag-green">Tier 1: Random Forest (18 Features)</span>
        <span class="tag-badge tag-blue">Tier 2: 3-Layer MLP (18 Features)</span>
        <span class="tag-badge tag-pink">Tier 3: 1D CNN (Raw Baseband I/Q [2, 128])</span>
        <span class="tag-badge tag-purple">15 Calibrated SNR Points (-10 to +18 dB)</span>
        <span class="tag-badge tag-amber">96,000 Bursts Dataset Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 5. Sidebar Controls & Test Burst Selection
# -------------------------------------------------------------
st.sidebar.markdown("### Bảng Điều Khiển Tín Hiệu")

if is_full_data:
    st.sidebar.success(f"Đang kết nối tập test đầy đủ: **{len(df_test):,} bursts**")
else:
    st.sidebar.info(f"Đang kết nối tập mẫu chuẩn: **{len(df_test):,} bursts**")

# A. Lọc theo Kênh truyền (Channel: AWGN vs Rayleigh3Tap)
channel_options = ["Tất cả kênh"] + sorted(df_test["channel"].dropna().unique().tolist())
selected_channel_filter = st.sidebar.selectbox(
    "Kênh truyền (Channel):",
    options=channel_options,
    index=0,
    help="So sánh tín hiệu chỉ có nhiễu Gauss (AWGN) với tín hiệu chịu pha-đinh đa đường (Rayleigh3Tap)"
)

# B. Lọc theo Modulation
mod_filter_options = ["Tất cả loại sóng"] + classes
selected_mod_filter = st.sidebar.selectbox(
    "Loại điều chế (Modulation Scheme):",
    options=mod_filter_options,
    index=0
)

# C. Lọc theo Calibrated SNR
snr_options = sorted(int(x) for x in df_test["snr_db"].unique())

st.sidebar.markdown("#### Chọn nhanh mức SNR đặc trưng:")
col_snr1, col_snr2, col_snr3 = st.sidebar.columns(3)
if col_snr1.button("-4 dB", use_container_width=True, help="Low SNR: Vùng nhiễu AWGN chi phối"):
    st.session_state.selected_snr = -4
if col_snr2.button("+4 dB", use_container_width=True, help="Mid SNR: Vùng chuyển tiếp pha-đinh Rayleigh"):
    st.session_state.selected_snr = 4
if col_snr3.button("+14 dB", use_container_width=True, help="High SNR: Tiệm cận giới hạn đặc trưng"):
    st.session_state.selected_snr = 14

if "selected_snr" not in st.session_state or st.session_state.selected_snr not in snr_options:
    st.session_state.selected_snr = 4

selected_snr = st.sidebar.select_slider(
    "Tỷ số tín hiệu trên nhiễu (SNR in dB):",
    options=snr_options,
    key="selected_snr"
)

# D. Lọc tập mẫu theo tất cả tiêu chí
filtered_mask = (df_test["snr_db"] == selected_snr)
if selected_channel_filter != "Tất cả kênh":
    filtered_mask = filtered_mask & (df_test["channel"] == selected_channel_filter)
if selected_mod_filter != "Tất cả loại sóng":
    filtered_mask = filtered_mask & (df_test["modulation"] == selected_mod_filter)

filtered_indices = df_test.index[filtered_mask].tolist()

if len(filtered_indices) == 0:
    st.warning("Không có mẫu kiểm thử nào thỏa mãn bộ lọc hiện tại.")
    st.stop()

# E. Chọn burst mẫu
if "burst_choice_idx" not in st.session_state or st.session_state.burst_choice_idx >= len(filtered_indices):
    st.session_state.burst_choice_idx = 0

if st.sidebar.button("Bốc mẫu ngẫu nhiên (Random Burst)", use_container_width=True):
    st.session_state.burst_choice_idx = int(np.random.randint(0, len(filtered_indices)))

burst_choice_idx = st.sidebar.slider(
    f"Mẫu trong nhóm lọc ({len(filtered_indices):,} bursts):",
    0,
    max(0, len(filtered_indices) - 1),
    st.session_state.burst_choice_idx
)
st.session_state.burst_choice_idx = burst_choice_idx

global_idx = filtered_indices[burst_choice_idx]
current_sample = df_test.iloc[global_idx]
true_label = current_sample["modulation"]
current_iq = np.array(iq_bursts[global_idx], dtype=np.float32)  # shape [2, 128]

st.sidebar.markdown("---")
st.sidebar.markdown("### Cấu hình hiển thị mô hình:")
show_rf = st.sidebar.checkbox("Tier 1: Random Forest (Features)", value=True)
show_mlp = st.sidebar.checkbox("Tier 2: 3-Layer MLP (Features)", value=True)
show_cnn = st.sidebar.checkbox("Tier 3: 1D CNN (Raw I/Q)", value=True)

# -------------------------------------------------------------
# 6. Main Tabs Interface (Now with 5 Comprehensive Tabs)
# -------------------------------------------------------------
tab_live, tab_bench, tab_whatif, tab_data, tab_theory = st.tabs([
    "So Sánh Suy Luận Trực Tiếp (Live Inference)",
    "Thư Viện Benchmark Toàn Diện (15 Mức SNR)",
    "Thử Nghiệm Tín Hiệu Tự Do (What-If Synthesizer)",
    "Kiến Trúc Bộ Dữ Liệu (96,000 Bursts Explorer)",
    "Cơ Sở Toán Học & Kỹ Thuật Vô Tuyến (RF Mechanics)"
])

# -------------------------------------------------------------
# TAB 1: LIVE INFERENCE SIDE-BY-SIDE
# -------------------------------------------------------------
with tab_live:
    # 1. Trích xuất đặc trưng & Chuẩn hóa
    x_raw = current_sample[feature_cols].values.astype(np.float64).reshape(1, -1)
    x_scaled = scaler.transform(x_raw)
    
    # Tier 1: Random Forest Inference
    t0 = time.perf_counter()
    rf_prob = rf_model.predict_proba(x_scaled)[0]
    t_rf = (time.perf_counter() - t0) * 1e6
    rf_pred_idx = int(np.argmax(rf_prob))
    rf_pred_label = classes[rf_pred_idx]
    rf_conf = rf_prob[rf_pred_idx] * 100
    rf_ok = (rf_pred_label == true_label)
    
    # Tier 2: MLP Inference
    x_tensor = torch.from_numpy(x_scaled.astype(np.float32))
    t0 = time.perf_counter()
    with torch.no_grad():
        mlp_logits = mlp_model(x_tensor)
        mlp_prob = torch.softmax(mlp_logits, dim=1).numpy()[0]
    t_mlp = (time.perf_counter() - t0) * 1e6
    mlp_pred_idx = int(np.argmax(mlp_prob))
    mlp_pred_label = classes[mlp_pred_idx]
    mlp_conf = mlp_prob[mlp_pred_idx] * 100
    mlp_ok = (mlp_pred_label == true_label)

    # Tier 3: 1D CNN Inference (on Raw I/Q Baseband [2, 128])
    iq_tensor = torch.from_numpy(current_iq).unsqueeze(0)  # shape [1, 2, 128]
    t0 = time.perf_counter()
    with torch.no_grad():
        cnn_logits = cnn_model(iq_tensor)
        cnn_prob = torch.softmax(cnn_logits, dim=1).numpy()[0]
    t_cnn = (time.perf_counter() - t0) * 1e6
    cnn_pred_idx = int(np.argmax(cnn_prob))
    cnn_pred_label = classes[cnn_pred_idx]
    cnn_conf = cnn_prob[cnn_pred_idx] * 100
    cnn_ok = (cnn_pred_label == true_label)

    # Banner thông tin gói sóng (Ground Truth)
    channel_badge_color = "#0284C7" if current_sample.get('channel') == "AWGN" else "#D97706"
    st.markdown(f"""
    <div class="ground-truth-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span class="stat-label">Loại sóng thực tế (Ground Truth):</span>
                <span style="font-size: 1.5rem; font-weight: 800; color: #0284C7; margin-left: 8px;">{true_label}</span>
                <span style="margin-left: 20px; font-weight: 600; color: #475569;">Mức SNR: <b>{selected_snr:+d} dB</b></span>
                <span style="margin-left: 15px; font-weight: 600; color: {channel_badge_color};">Kênh: <b>{current_sample.get('channel', 'Rayleigh3Tap')}</b></span>
                <span style="margin-left: 15px; color: #64748B;">| Mã mẫu: <code>{current_sample.get('example_id', f'Burst #{global_idx}')}</code></span>
            </div>
            <div style="font-size: 0.85rem; color: #64748B;">
                Chuẩn hóa RMS: <b>r[n]/E_rms</b> • Chiều dài: <b>L = 128 samples</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3 Cột so sánh 3 mô hình side-by-side
    col_rf, col_mlp, col_cnn = st.columns(3)
    
    with col_rf:
        badge = '<span class="badge-correct">CHÍNH XÁC</span>' if rf_ok else '<span class="badge-wrong">NHẦM LẪN</span>'
        color_val = "#16A34A" if rf_ok else "#DC2626"
        st.markdown(f"""
        <div class="model-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="stat-label">Tier 1: Classical ML</span>
                {badge}
            </div>
            <h3 style="margin: 8px 0 2px 0; font-size: 1.25rem; color: #10B981;">Random Forest</h3>
            <p style="font-size: 0.78rem; color: #64748B; margin-bottom: 12px;">18 Domain Features • 100 Trees (~1 GB)</p>
            <hr style="margin: 8px 0; border: 0; border-top: 1px solid #F1F5F9;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 10px;">
                <span style="font-size: 0.85rem; color: #64748B;">Dự đoán:</span>
                <span style="font-size: 1.55rem; font-weight: 800; color: {color_val};">{rf_pred_label}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="font-size: 0.85rem; color: #64748B;">Độ tin cậy:</span>
                <span style="font-size: 1.15rem; font-weight: 700; color: #334155;">{rf_conf:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
                <span style="font-size: 0.85rem; color: #64748B;">Độ trễ (Latency):</span>
                <span style="font-size: 0.95rem; font-weight: 600; color: #8B5CF6;">{t_rf:.1f} µs</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_mlp:
        badge = '<span class="badge-correct">CHÍNH XÁC</span>' if mlp_ok else '<span class="badge-wrong">NHẦM LẪN</span>'
        color_val = "#16A34A" if mlp_ok else "#DC2626"
        st.markdown(f"""
        <div class="model-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="stat-label">Tier 2: Neural Network</span>
                {badge}
            </div>
            <h3 style="margin: 8px 0 2px 0; font-size: 1.25rem; color: #3B82F6;">3-Layer MLP</h3>
            <p style="font-size: 0.78rem; color: #64748B; margin-bottom: 12px;">18 Domain Features • 11,592 params (0.05 MB)</p>
            <hr style="margin: 8px 0; border: 0; border-top: 1px solid #F1F5F9;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 10px;">
                <span style="font-size: 0.85rem; color: #64748B;">Dự đoán:</span>
                <span style="font-size: 1.55rem; font-weight: 800; color: {color_val};">{mlp_pred_label}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="font-size: 0.85rem; color: #64748B;">Độ tin cậy:</span>
                <span style="font-size: 1.15rem; font-weight: 700; color: #334155;">{mlp_conf:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
                <span style="font-size: 0.85rem; color: #64748B;">Độ trễ (Latency):</span>
                <span style="font-size: 0.95rem; font-weight: 600; color: #8B5CF6;">{t_mlp:.1f} µs</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_cnn:
        badge = '<span class="badge-correct">CHÍNH XÁC</span>' if cnn_ok else '<span class="badge-wrong">NHẦM LẪN</span>'
        color_val = "#16A34A" if cnn_ok else "#DC2626"
        st.markdown(f"""
        <div class="model-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="stat-label">Tier 3: Deep ConvNet</span>
                {badge}
            </div>
            <h3 style="margin: 8px 0 2px 0; font-size: 1.25rem; color: #EC4899;">1D CNN</h3>
            <p style="font-size: 0.78rem; color: #64748B; margin-bottom: 12px;">Raw Baseband I/Q [2, 128] • 53,224 params (0.21 MB)</p>
            <hr style="margin: 8px 0; border: 0; border-top: 1px solid #F1F5F9;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 10px;">
                <span style="font-size: 0.85rem; color: #64748B;">Dự đoán:</span>
                <span style="font-size: 1.55rem; font-weight: 800; color: {color_val};">{cnn_pred_label}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="font-size: 0.85rem; color: #64748B;">Độ tin cậy:</span>
                <span style="font-size: 1.15rem; font-weight: 700; color: #334155;">{cnn_conf:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
                <span style="font-size: 0.85rem; color: #64748B;">Độ trễ (Latency):</span>
                <span style="font-size: 0.95rem; font-weight: 600; color: #8B5CF6;">{t_cnn:.1f} µs</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Phân phối xác suất 8 lớp & Biểu diễn tín hiệu I/Q thời gian thực
    col_chart_prob, col_signal = st.columns([1.3, 1])

    with col_chart_prob:
        st.subheader("So Sánh Phân Phối Xác Suất 8 Lớp (Class Probabilities)")
        
        df_prob_comp = pd.DataFrame({
            "Modulation": classes,
            "Random Forest": rf_prob * 100,
            "3-Layer MLP": mlp_prob * 100,
            "1D CNN": cnn_prob * 100
        })
        
        fig_prob, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
        x_indices = np.arange(len(classes))
        width = 0.26
        
        ax.bar(x_indices - width, df_prob_comp["Random Forest"], width=width, label="Tier 1: Random Forest", color="#10B981", alpha=0.9)
        ax.bar(x_indices, df_prob_comp["3-Layer MLP"], width=width, label="Tier 2: 3-Layer MLP", color="#3B82F6", alpha=0.9)
        ax.bar(x_indices + width, df_prob_comp["1D CNN"], width=width, label="Tier 3: 1D CNN", color="#EC4899", alpha=0.9)
        
        # Đánh dấu ground truth
        for idx, mod in enumerate(classes):
            if mod == true_label:
                ax.axvspan(idx - 0.45, idx + 0.45, color="#FEF08A", alpha=0.28, zorder=0)
                ax.text(idx, 102, "TRUE", ha='center', va='bottom', fontsize=8, fontweight='bold', color="#CA8A04")

        ax.set_xticks(x_indices)
        ax.set_xticklabels(classes, rotation=25, ha='right', fontsize=9, fontweight='600')
        ax.set_ylabel("Xác suất dự đoán (%)", fontsize=10, fontweight='bold')
        ax.set_ylim(0, 115)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_prob)
        plt.close()

    with col_signal:
        st.subheader("Không Gian Tín Hiệu I/Q & Dạng Sóng Thời Gian Thực")
        
        fig_sig, (ax_const, ax_wave) = plt.subplots(2, 1, figsize=(6.5, 4.8), dpi=150)
        
        # 1. Chòm sao I/Q
        I_samples = current_iq[0]
        Q_samples = current_iq[1]
        ax_const.scatter(I_samples, Q_samples, color="#0284C7", alpha=0.75, s=22, edgecolors="none")
        ax_const.axhline(0, color="gray", linestyle=":", alpha=0.6)
        ax_const.axvline(0, color="gray", linestyle=":", alpha=0.6)
        ax_const.set_title(f"I/Q Constellation (L=128 Samples)", fontsize=9.5, fontweight='bold', pad=4)
        ax_const.set_xlabel("In-Phase I[n]", fontsize=8)
        ax_const.set_ylabel("Quadrature Q[n]", fontsize=8)
        ax_const.set_xlim(-3, 3)
        ax_const.set_ylim(-3, 3)
        ax_const.grid(True, linestyle="--", alpha=0.4)

        # 2. Dạng sóng thời gian
        time_axis = np.arange(128)
        ax_wave.plot(time_axis, I_samples, label="I[n]", color="#3B82F6", linewidth=1.2)
        ax_wave.plot(time_axis, Q_samples, label="Q[n]", color="#EC4899", linewidth=1.2, alpha=0.85)
        ax_wave.set_title("Baseband Waveform Trajectory [2, 128]", fontsize=9.5, fontweight='bold', pad=4)
        ax_wave.set_xlabel("Sample Index n (0..127)", fontsize=8)
        ax_wave.set_ylabel("Normalized Amplitude", fontsize=8)
        ax_wave.legend(fontsize=7.5, loc="upper right")
        ax_wave.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        st.pyplot(fig_sig)
        plt.close()

    # Dải Profile 18 đặc trưng bên dưới
    st.markdown("---")
    st.subheader("Profile 18 Đặc Trưng Vật Lý Của Gói Sóng (Z-Score Profile)")
    feat_vals = x_scaled[0]
    df_feat = pd.DataFrame({
        "Feature": feature_cols,
        "Z-Score": feat_vals
    })
    
    fig_feat, ax_f = plt.subplots(figsize=(12, 3.2), dpi=150)
    colors_feat = ['#EF4444' if f == 'C40' else ('#8B5CF6' if 'C' in f else '#0284C7') for f in feature_cols]
    ax_f.bar(df_feat["Feature"], df_feat["Z-Score"], color=colors_feat, width=0.55)
    ax_f.set_xticks(range(len(feature_cols)))
    ax_f.set_xticklabels(feature_cols, rotation=45, ha='right', fontsize=8.5, fontweight='600')
    ax_f.set_ylabel("Z-Score", fontsize=9, fontweight='bold')
    ax_f.grid(axis='y', linestyle='--', alpha=0.5)
    ax_f.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_feat)
    plt.close()
    st.caption("C40 (Cumulant bậc 4: Phân biệt BPSK & QPSK) | Cumulants bậc 2..6 (Bậc cao) | Thống kê phổ & Công suất")

# -------------------------------------------------------------
# TAB 2: COMPREHENSIVE BENCHMARK GALLERY
# -------------------------------------------------------------
with tab_bench:
    st.markdown("### Báo Cáo Thực Nghiệm Toàn Diện (15 Điểm Calibrated SNR: -10 dB đến +18 dB)")
    st.write("Đánh giá hiệu năng 3 tầng mô hình: **Random Forest** (Tier 1), **3-Layer MLP** (Tier 2), và **1D CNN** (Tier 3).")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("**Đường cong Accuracy vs. SNR (-10 dB đến +18 dB):**")
        p_acc = os.path.join(ASSETS_DIR, "benchmark_accuracy_vs_snr.png")
        if os.path.exists(p_acc):
            st.image(p_acc, use_container_width=True)
    with col_c2:
        st.markdown("**Đường cong Macro-F1 vs. SNR (-10 dB đến +18 dB):**")
        p_f1 = os.path.join(ASSETS_DIR, "benchmark_macro_f1_vs_snr.png")
        if os.path.exists(p_f1):
            st.image(p_f1, use_container_width=True)
            
    st.markdown("---")
    st.markdown("### Ma Trận Nhầm Lẫn (Confusion Matrices) So Sánh 3 Tầng Mô Hình")
    st.write("Quan sát sự phục hồi phân loại từ vùng nhiễu AWGN (-4 dB) đến chuyển tiếp (+4 dB) và bứt phá tiệm cận (+14 dB).")
    
    cm_tabs = st.tabs(["Low SNR (-4 dB): Vùng nhiễu AWGN chi phối", "Mid SNR (+4 dB): Vùng chuyển tiếp pha", "High SNR (+14 dB): Vùng tiệm cận trần"])
    
    with cm_tabs[0]:
        p_cm_low = os.path.join(ASSETS_DIR, "cm_comparison_-4dB.png")
        if os.path.exists(p_cm_low):
            st.image(p_cm_low, caption="Ma trận nhầm lẫn ở SNR = -4 dB (Random Forest vs 3-Layer MLP vs 1D CNN)", use_container_width=True)
        st.info("**Nhận xét ở -4 dB:** Nhiễu AWGN lớn làm méo dạng hàm mật độ xác suất, các cumulant mẫu bị biến dạng mạnh. Cả 3 mô hình đều bị nhầm lẫn giữa các họ điều chế.")
        
    with cm_tabs[1]:
        p_cm_mid = os.path.join(ASSETS_DIR, "cm_comparison_+4dB.png")
        if os.path.exists(p_cm_mid):
            st.image(p_cm_mid, caption="Ma trận nhầm lẫn ở SNR = +4 dB (Random Forest vs 3-Layer MLP vs 1D CNN)", use_container_width=True)
        st.info("**Nhận xét ở +4 dB:** BPSK được nhận dạng gần như hoàn hảo (~98%), QPSK tách rõ (~57%). Mô hình 1D CNN bắt đầu vượt trội nhờ khả năng bám quỹ đạo pha thời gian thực.")

    with cm_tabs[2]:
        p_cm_high = os.path.join(ASSETS_DIR, "cm_comparison_+14dB.png")
        if os.path.exists(p_cm_high):
            st.image(p_cm_high, caption="Ma trận nhầm lẫn ở SNR = +14 dB (Random Forest vs 3-Layer MLP vs 1D CNN)", use_container_width=True)
        st.info("**Nhận xét ở +14 dB:** Cả RF và MLP đều chạm trần toán học (~48.25%) do các cumulant bậc <= 6 của (8PSK, 16PSK, 32PSK) và (16QAM, 64QAM, 256QAM) bị nhầm lẫn nội bộ. Trong khi đó, **1D CNN đạt 53.12% – 55.10%** nhờ học trực tiếp từ biểu diễn chuỗi thời gian của chòm sao I/Q!")

    st.markdown("---")
    st.markdown("### Bảng So Sánh Chỉ Số Kỹ Thuật Tổng Hợp (Executive Summary)")
    sum_csv = os.path.join(ASSETS_DIR, "comprehensive_models_executive_summary.csv")
    if os.path.exists(sum_csv):
        df_exec_all = pd.read_csv(sum_csv)
        st.dataframe(df_exec_all, use_container_width=True, hide_index=True)
        
    if df_bench is not None:
        st.markdown("#### Dữ liệu chi tiết 15 điểm Calibrated SNR:")
        st.dataframe(df_bench, use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 3: INTERACTIVE WHAT-IF SYNTHESIZER
# -------------------------------------------------------------
with tab_whatif:
    st.markdown("### Thử Nghiệm Tín Hiệu Tự Do (What-If Arbitrary Burst Synthesizer)")
    st.write("Trực tiếp điều chỉnh các tham số đặc trưng vật lý quan trọng (Cumulants, PAPR, Spectral) để quan sát phản hồi tức thì từ các mô hình.")
    
    st.markdown("#### Cấu hình tín hiệu mẫu định sẵn (Presets):")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    preset_clicked = None
    if pcol1.button("Giả lập BPSK lý tưởng", use_container_width=True):
        preset_clicked = "BPSK"
    if pcol2.button("Giả lập QPSK lý tưởng", use_container_width=True):
        preset_clicked = "QPSK"
    if pcol3.button("Giả lập QAM đỉnh cao (High PAPR)", use_container_width=True):
        preset_clicked = "QAM"
    if pcol4.button("Giả lập Nhiễu đa đường nặng", use_container_width=True):
        preset_clicked = "NOISY"
        
    base_feats = current_sample[feature_cols].copy()
    if preset_clicked == "BPSK":
        base_feats["C40"] = -2.0
        base_feats["C20"] = 1.0
        base_feats["C42"] = 1.0
        base_feats["PAPR"] = 0.5
    elif preset_clicked == "QPSK":
        base_feats["C40"] = -1.0
        base_feats["C20"] = 0.0
        base_feats["C42"] = 1.0
        base_feats["PAPR"] = 0.8
    elif preset_clicked == "QAM":
        base_feats["C40"] = -0.68
        base_feats["C20"] = 0.0
        base_feats["PAPR"] = 4.5
    elif preset_clicked == "NOISY":
        base_feats["C40"] = 0.05
        base_feats["C20"] = 0.02
        base_feats["H_phase"] = 3.5

    st.markdown("---")
    st.markdown("#### Tham số vật lý điều khiển tín hiệu:")
    scol1, scol2, scol3, scol4 = st.columns(4)
    with scol1:
        c40_val = st.slider("C40 (Cumulant bậc 4):", -2.5, 0.5, float(base_feats["C40"]), 0.05,
                            help="BPSK ~ -2.0, QPSK ~ -1.0, 16QAM ~ -0.68, 8PSK/16PSK ~ 0.0")
    with scol2:
        c20_val = st.slider("C20 (Cumulant bậc 2):", -0.5, 1.5, float(base_feats["C20"]), 0.05,
                            help="BPSK = 1.0, các điều chế pha tròn khác ~ 0.0")
    with scol3:
        papr_val = st.slider("PAPR (Peak-to-Average Power):", 0.0, 10.0, float(base_feats["PAPR"]), 0.2,
                             help="Tỷ số công suất đỉnh trên trung bình")
    with scol4:
        gamma_val = st.slider("gamma_max (Spectral density max):", 0.0, 15.0, float(base_feats["gamma_max"]), 0.2,
                              help="Công suất phổ cực đại chuẩn hóa")

    # Cập nhật vector đặc trưng
    synth_feats = base_feats.copy()
    synth_feats["C40"] = c40_val
    synth_feats["C20"] = c20_val
    synth_feats["PAPR"] = papr_val
    synth_feats["gamma_max"] = gamma_val

    # Chuẩn hóa & Suy luận
    x_synth_raw = synth_feats[feature_cols].values.astype(np.float64).reshape(1, -1)
    x_synth_scaled = scaler.transform(x_synth_raw)

    p_rf = rf_model.predict_proba(x_synth_scaled)[0]
    with torch.no_grad():
        p_mlp = torch.softmax(mlp_model(torch.from_numpy(x_synth_scaled.astype(np.float32))), dim=1).numpy()[0]
        # For CNN on synthetic: generate synthetic I/Q burst corresponding to key features
        t_steps = np.linspace(0, 4 * np.pi, 128)
        sim_I = np.cos(t_steps) * (1.0 + 0.1 * papr_val) + np.random.randn(128) * 0.1
        sim_Q = np.sin(t_steps) * (1.0 + 0.1 * papr_val) + np.random.randn(128) * 0.1
        sim_iq = np.stack([sim_I, sim_Q], axis=0).astype(np.float32)
        p_cnn = torch.softmax(cnn_model(torch.from_numpy(sim_iq).unsqueeze(0)), dim=1).numpy()[0]

    st.markdown("### Kết Quả Phân Loại Trực Tiếp Trên Tín Hiệu Giả Lập:")
    res_c1, res_c2, res_c3 = st.columns(3)
    
    with res_c1:
        idx = int(np.argmax(p_rf))
        st.metric("Tier 1: Random Forest", f"{classes[idx]}", f"Độ tin cậy: {p_rf[idx]*100:.1f}%")
    with res_c2:
        idx = int(np.argmax(p_mlp))
        st.metric("Tier 2: 3-Layer MLP", f"{classes[idx]}", f"Độ tin cậy: {p_mlp[idx]*100:.1f}%")
    with res_c3:
        idx = int(np.argmax(p_cnn))
        st.metric("Tier 3: 1D CNN", f"{classes[idx]}", f"Độ tin cậy: {p_cnn[idx]*100:.1f}%")

    # Biểu đồ so sánh
    fig_synth, ax_s = plt.subplots(figsize=(10, 3.8), dpi=150)
    x_idx = np.arange(len(classes))
    ax_s.bar(x_idx - 0.26, p_rf * 100, width=0.25, label="Random Forest", color="#10B981")
    ax_s.bar(x_idx, p_mlp * 100, width=0.25, label="3-Layer MLP", color="#3B82F6")
    ax_s.bar(x_idx + 0.26, p_cnn * 100, width=0.25, label="1D CNN", color="#EC4899")
    ax_s.set_xticks(x_idx)
    ax_s.set_xticklabels(classes, fontsize=9, fontweight='600')
    ax_s.set_ylabel("Xác suất (%)", fontsize=10)
    ax_s.set_ylim(0, 105)
    ax_s.grid(axis='y', linestyle='--', alpha=0.5)
    ax_s.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_synth)
    plt.close()

# -------------------------------------------------------------
# TAB 4: DATASET ARCHITECTURE & EXPLORER (96,000 BURSTS)
# -------------------------------------------------------------
with tab_data:
    st.markdown("### Kiến Trúc Toàn Bộ Dữ Liệu (96,000 Bursts AMC Dataset)")
    st.write("Tổng quan bộ dữ liệu đã sinh ra, phân bổ tập Train/Val/Test và đặc tả kỹ thuật kênh truyền.")
    
    # Cards thống kê tổng quan
    dc1, dc2, dc3, dc4 = st.columns(4)
    with dc1:
        st.markdown("""
        <div class="dataset-metric-card">
            <div style="font-size: 0.8rem; color: #64748B; font-weight: 700;">TỔNG SỐ LƯỢNG GÓI SÓNG</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #0284C7;">96,000</div>
            <div style="font-size: 0.75rem; color: #10B981; font-weight: 600;">L = 128 mẫu I/Q mỗi burst</div>
        </div>
        """, unsafe_allow_html=True)
    with dc2:
        st.markdown("""
        <div class="dataset-metric-card">
            <div style="font-size: 0.8rem; color: #64748B; font-weight: 700;">TẬP HUẤN LUYỆN (TRAIN SPLIT)</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #10B981;">67,200 (70%)</div>
            <div style="font-size: 0.75rem; color: #64748B;">280 bursts / tổ hợp cấu hình</div>
        </div>
        """, unsafe_allow_html=True)
    with dc3:
        st.markdown("""
        <div class="dataset-metric-card">
            <div style="font-size: 0.8rem; color: #64748B; font-weight: 700;">TẬP KIỂM CHỨNG (VAL SPLIT)</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #F59E0B;">14,400 (15%)</div>
            <div style="font-size: 0.75rem; color: #64748B;">60 bursts / tổ hợp cấu hình</div>
        </div>
        """, unsafe_allow_html=True)
    with dc4:
        st.markdown("""
        <div class="dataset-metric-card">
            <div style="font-size: 0.8rem; color: #64748B; font-weight: 700;">TẬP KIỂM THỬ (TEST SPLIT)</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #8B5CF6;">14,400 (15%)</div>
            <div style="font-size: 0.75rem; color: #64748B;">60 bursts / tổ hợp cấu hình</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_d_left, col_d_right = st.columns(2)
    with col_d_left:
        st.markdown("#### 1. Cấu Trúc Đa Dạng Điều Chế & Mức Nhiễu:")
        st.markdown("""
        * **8 Loại điều chế (Modulations):**
          * **Họ PSK:** `BPSK`, `QPSK`, `8PSK`, `16PSK`, `32PSK` (điều chế dịch pha).
          * **Họ QAM:** `16QAM`, `64QAM`, `256QAM` (điều chế biên độ - pha vuông góc).
        * **15 Mức SNR Calibrated Grid:**
          $$\\text{SNR} \\in \\{-10, -8, -6, -4, -2, 0, +2, +4, +6, +8, +10, +12, +14, +16, +18\\}\\text{ dB}$$
        * **2 Mô hình Kênh truyền vô tuyến:**
          * **AWGN:** Nhiễu Gauss trắng cộng thuần túy.
          * **Rayleigh3Tap:** Kênh pha-đinh đa đường 3-tap gây xoay pha và tán xạ cụm chòm sao.
        """)
        
    with col_d_right:
        st.markdown("#### 2. Định Dạng Lưu Trữ Khớp 100% Theo Index:")
        st.markdown("""
        * **Đối với Classical ML & MLP:**
          * File `features_train.csv`, `features_val.csv`, `features_test.csv` (18 cột đặc trưng $C_{20} \\dots H_{psd}$).
        * **Đối với 1D CNN:**
          * File `iq_train.npz`, `iq_val.npz`, `iq_test.npz` (chứa ma trận tensor $N \\times 2 \\times 128$).
        * **Bảo đảm Zero Data Leakage:**
          * Cắt phân vùng cứng (Rigid Partitioning 70/15/15).
          * Không có gói sóng test nào dùng chung seed ngẫu nhiên hoặc hiện thực hóa kênh với tập train.
          * Cột metadata (`snr_db`, `channel`) được cách ly hoàn toàn khỏi input của mô hình.
        """)

    st.markdown("---")
    st.markdown("#### 3. Bảng Phân Bổ 18 Đặc Trưng Vật Lý (Domain Features):")
    feat_meta = [
        {"Nhóm": "Nhóm A: Cumulants Bậc Cao", "Tên đặc trưng": "C20, C21", "Ý nghĩa vật lý": "Cumulants bậc 2; C21 đại diện cho công suất chuẩn hóa (= 1.0)"},
        {"Nhóm": "Nhóm A: Cumulants Bậc Cao", "Tên đặc trưng": "C40, C41, C42", "Ý nghĩa vật lý": "Cumulants bậc 4; C40 tách biệt BPSK (-2.0), QPSK (-1.0), QAM (-0.68)"},
        {"Nhóm": "Nhóm A: Cumulants Bậc Cao", "Tên đặc trưng": "C60, C61, C62, C63", "Ý nghĩa vật lý": "Cumulants bậc 6; phân tích hình học chòm sao đa trạng thái"},
        {"Nhóm": "Nhóm B: Phổ & Thống Kê", "Tên đặc trưng": "sigma_aa", "Ý nghĩa vật lý": "Độ lệch chuẩn biên độ tín hiệu"},
        {"Nhóm": "Nhóm B: Phổ & Thống Kê", "Tên đặc trưng": "H_phase", "Ý nghĩa vật lý": "Entropy của histogram pha (đo độ phân tán góc pha)"},
        {"Nhóm": "Nhóm B: Phổ & Thống Kê", "Tên đặc trưng": "sigma_ap, sigma_dp", "Ý nghĩa vật lý": "Độ lệch chuẩn của pha tuyệt đối và pha vi sai"},
        {"Nhóm": "Nhóm B: Phổ & Thống Kê", "Tên đặc trưng": "PAPR", "Ý nghĩa vật lý": "Peak-to-Average Power Ratio (rất cao ở 64QAM/256QAM, thấp ở PSK)"},
        {"Nhóm": "Nhóm B: Phổ & Thống Kê", "Tên đặc trưng": "gamma_max, Kurt_psd, Skew_psd, H_psd", "Ý nghĩa vật lý": "Các chỉ số mật độ phổ công suất PSD tính bằng phương pháp Welch"},
    ]
    st.dataframe(pd.DataFrame(feat_meta), use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 5: DEEP PHYSICAL RF MECHANICS & MATH
# -------------------------------------------------------------
with tab_theory:
    st.markdown("### Cơ Sở Toán Học & Cơ Chế Vật Lý Vô Tuyến (Baseband RF Mechanics)")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("""
        #### 1. Cơ Chế Tín Hiệu I/Q & Chuẩn Hóa Năng Lượng
        * **Độ dài gói tín hiệu:** $L = 128$ mẫu baseband phức mỗi burst.
        * **Khuôn dạng Tensor:** Dữ liệu có kích thước thực $[N_{batch}, 2, 128]$ với Hàng 0 là In-phase $I[n]$ và Hàng 1 là Quadrature $Q[n]$.
        * **Chuẩn hóa RMS (No Energy Cheating):**
          $$r_{norm}[n] = \\frac{r[n]}{\\sqrt{\\frac{1}{L} \\sum_{k=0}^{L-1} |r[k]|^2}}$$
          Ngăn chặn mô hình phân loại dựa vào công suất trung bình thay vì cấu trúc hình học chòm sao.
        
        #### 2. Sức Mạnh Của Cumulant Bậc Cao ($C_{40}, C_{42}$)
        * Với tín hiệu chuẩn hóa công suất:
          * **BPSK:** $C_{40} = -2.0$
          * **QPSK:** $C_{40} = -1.0$
          * **16QAM:** $C_{40} \\approx -0.68$
          * **64QAM:** $C_{40} \\approx -0.62$
          * **$M$-PSK ($M \\ge 8$):** $C_{40} = 0.0$
        """)
    with col_t2:
        st.markdown("""
        #### 3. Vì Sao 1D CNN Vượt Qua Giới Hạn Tiệm Cận (~48.25%)?
        * Ở cumulant bậc $\\le 6$, các điều chế $M$-PSK bậc cao (`8PSK`, `16PSK`, `32PSK`) có chung đặc tính lý thuyết:
          $$|C_{40}| = 0, \\quad |C_{42}| = 1, \\quad |C_{60}| = 0$$
          Do bộ 18 đặc trưng dừng ở bậc 6, Random Forest và MLP bị kẹt ở trần lý thuyết $\\approx 48.25\%$.
        * **Đột phá của 1D CNN:**
          * Mô hình 1D CNN nhận trực tiếp chuỗi tín hiệu thời gian thực $[2, 128]$.
          * Các bộ lọc tích chập (Conv1d) với kích thước kernel $K=5$ học được mối tương quan pha liên tục (continuous phase trajectories) giữa các mẫu kế tiếp nhau.
          * Nhờ đó, 1D CNN đạt **$44.00\%$ tổng thể** và vươn tới **$55.10\%$ ở SNR cao**, vượt qua trần phân biệt của bộ đặc trưng truyền thống!
        """)
        
    st.markdown("---")
    st.markdown("""
    #### 4. Phân Tích Đánh Đổi Kỹ Thuật (Engineering Trade-offs):
    | Tiêu chí | Tier 1: Random Forest | Tier 2: 3-Layer MLP | Tier 3: 1D CNN |
    | :--- | :--- | :--- | :--- |
    | **Đầu vào** | 18 Đặc trưng thủ công | 18 Đặc trưng chuẩn hóa | **Raw Baseband I/Q [2, 128]** |
    | **Kích thước mô hình** | Rất lớn (~1,000 MB do 100 cây sâu) | Cực nhỏ (**0.05 MB**, 11,592 params) | Tinh gọn (**0.21 MB**, 53,224 params) |
    | **Độ trễ suy luận đơn ($B=1$)** | ~280 µs | **Siêu nhanh (~36 µs)** | ~480 µs |
    | **Độ trễ xử lý theo lô (Batched)** | 177 µs / mẫu | **0.71 – 55 µs / mẫu** | 58 µs / mẫu |
    | **Độ chính xác tổng thể** | 33.83% | 35.17% | **44.00% (Cao nhất)** |
    | **Triển khai thực tế (Edge / FPGA)** | Khó khả thi do bộ nhớ 1 GB | Rất tốt cho MCU/DSP tiết kiệm năng lượng | **Lý tưởng cho phần cứng xử lý sóng SDR/FPGA** |
    """)

# -------------------------------------------------------------
# Footer
# -------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 10px 0;">
    <b>EE136IU Capstone Mini-Project • Topic 1: Comprehensive Model Benchmark</b><br>
    Classical ML (Random Forest) vs. 3-Layer MLP vs. 1D CNN • 96,000 Bursts AMC Dataset • AWGN & Rayleigh Fading Channels
</div>
""", unsafe_allow_html=True)
