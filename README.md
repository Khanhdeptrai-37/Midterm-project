# EE136: Machine Learning for Wireless Communications
## Topic 1: Comprehensive Model Benchmark — Classical ML vs. 3-Layer MLP vs. 1D CNN

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Automatic Modulation Classification (AMC) across 8 digital modulation schemes** under AWGN and Rayleigh 3-Tap multipath fading channels (-10 dB to +18 dB SNR). Evaluates three model tiers spanning hand-crafted domain features to raw time-domain baseband tensor representations.

---

## Executive Benchmark Summary

| Engineering Metric | Tier 1: Classical ML (Random Forest) | Tier 2: Neural Net (3-Layer MLP) | Tier 3: Deep ConvNet (1D CNN) | Hardware & Deployment Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Input Representation** | 18 Domain Features | 18 Normalized Features | **Raw Baseband I/Q $[2, 128]$** | 1D CNN requires zero manual feature extraction |
| **Overall Test Accuracy** | 33.83% | 35.17% | **44.00% (Highest Overall)** | 1D CNN outperforms by +8.8% to +10.2% |
| **Peak High-SNR Accuracy (+18 dB)** | 50.10% | 48.25% (Feature Ceiling) | **55.10% (Breaks Ceiling!)** | Breaks theoretical cumulant order 6 limit |
| **Macro-F1 Score** | 33.48% | 34.82% | **35.97% (Most Balanced)** | Highly balanced across all 8 classes |
| **Trainable Parameters** | 140,572,728 tree nodes | **11,592 parameters** | 53,224 parameters | Random Forest suffers parameter explosion |
| **Model File Size** | 999.7 MB (~1 GB) | **52.5 KB (Ultra Compact)** | 0.213 MB (218 KB) | MLP is 19,000x lighter than Random Forest |
| **Single-Burst Latency ($B=1$)** | ~280 µs | **~36 µs (Ultra Fast)** | ~424 µs | All 3 satisfy 5G NR frame budget ($< 1.0$ ms) |
| **Target Hardware Platform** | Cloud / Baseband Server | **Edge IoT / MCU / Low-Power DSP** | **SDR / FPGA Baseband Accelerator** | Flexible deployment matching hardware constraints |

---

## Repository Structure

```plaintext
├── app.py                                  # Unified Web Dashboard (Vietnamese Edition)
├── app_en.py                               # Unified Web Dashboard (English Academic Edition)
├── run_dashboard.bat                       # 1-Click Local Launcher (Vietnamese)
├── run_dashboard_en.bat                    # 1-Click Local Launcher (English)
├── run_public_online.bat                   # 1-Click Public Online Tunnel Launcher
├── evaluate_all.py                         # Standalone Comprehensive Evaluator (3 Tiers)
├── ML.ipynb                                # Classical ML Training & Analysis Notebook (RF & SVM)
├── requirements.txt                        # Python Dependencies
├── .gitignore                              # Git exclusion rules (>100MB files, binaries)
│
├── MLP/                                    # Tier 2: 3-Layer MLP Package
│   ├── best_mlp.pth                        # Pre-trained MLP Checkpoint (52.5 KB)
│   ├── model.py                            # PyTorch Architecture Definition (11,592 params)
│   └── mlp.ipynb                           # Training & Verification Notebook
│
├── EE136_1D_CNN/                           # Tier 3: 1D CNN Package
│   ├── cnn1d_best.pt                       # Pre-trained 1D CNN Checkpoint (53,224 params)
│   └── EE136_1D_CNN.ipynb                  # Training & Verification Notebook
│
├── ML/                                     # Tier 1: Classical Machine Learning Package
│   ├── scaler.pkl                          # Pre-fitted StandardScaler on 67.2k train bursts
│   └── label_encoder.pkl                   # 8-Class Label Encoder
│
├── assets/                                 # Visual & Numerical Deliverables
│   ├── benchmark_accuracy_vs_snr.png       # Comparative Accuracy Curve across 15 SNRs
│   ├── benchmark_macro_f1_vs_snr.png       # Comparative Macro-F1 Curve across 15 SNRs
│   ├── cm_comparison_-4dB.png              # Side-by-side Confusion Matrices at Low SNR
│   ├── cm_comparison_+4dB.png              # Side-by-side Confusion Matrices at Mid SNR
│   ├── cm_comparison_+14dB.png             # Side-by-side Confusion Matrices at High SNR
│   ├── sample_test_iq.npz                  # Pre-cached Representative Test Bursts (Runs Out-of-the-Box)
│   ├── comprehensive_models_executive_summary.csv
│   └── models_benchmark_comparison_3tier.csv
│
└── presentations/ (or root)
    ├── EE136IU_Topic1_Complete_Benchmark_Presentation.pptx  # Master 20-Slide Presentation
    └── EE136IU_Topic1_Classical_ML_Presentation.pptx        # Standalone Classical ML Deck
```

---

## Quickstart Guide

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/<your-username>/EE136-AMC-Model-Benchmark.git
cd EE136-AMC-Model-Benchmark

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the Interactive Web Dashboard

Choose your preferred edition:

* **English Edition (Academic / International)**:
  ```bash
  python -m streamlit run app_en.py
  # Or double-click: run_dashboard_en.bat
  ```
* **Vietnamese Edition**:
  ```bash
  python -m streamlit run app.py
  # Or double-click: run_dashboard.bat
  ```

The dashboard includes a pre-cached representative test dataset in `assets/sample_test_iq.npz`, allowing it to run immediately upon cloning without downloading gigabytes of raw data!

---

## Dataset Architecture (96,000 Bursts)

The complete dataset consists of 96,000 bursts of length $L=128$ complex samples ($N_{batch} \times 2 \times 128$), strictly partitioned into:
* **Train Split (70%)**: 67,200 bursts (280 bursts / condition)
* **Val Split (15%)**: 14,400 bursts (60 bursts / condition)
* **Test Split (15%)**: 14,400 bursts (60 bursts / condition)

**Operating Conditions**:
* **8 Modulations**: BPSK, QPSK, 8PSK, 16PSK, 32PSK, 16QAM, 64QAM, 256QAM.
* **15 Calibrated SNRs**: $-10\text{ dB}$ to $+18\text{ dB}$ (2 dB step).
* **2 Radio Channels**: AWGN (Thermal Noise) & Rayleigh 3-Tap Multipath Fading.

> *Note: Due to GitHub's 100 MB file limit, the full 1.2 GB dataset and the 999.7 MB uncompressed Random Forest model (`rf_model.pkl`) can be downloaded from [Releases / Cloud Storage](#).*

---

## Key Research Findings

1. **The Hand-Crafted Feature Ceiling (~48.25%)**:
   Both Random Forest and 3-Layer MLP plateau at high SNR due to mathematical cumulant degeneration: cumulants up to 6th order for $M \ge 8$ PSK collapse to $|C_{40}| = 0.0$ and $|C_{42}| = 1.0$.
2. **Deep Learning Temporal Advantage**:
   1D CNN directly consumes raw baseband tensors $[2, 128]$. Its 1D temporal kernels ($K=5$) learn continuous phase transitions and constellation trajectories, breaking through to **55.10%** at $+18\text{ dB}$ (+6.8% over feature-based models).
3. **Hardware Latency Trade-Off**:
   * **3-Layer MLP**: Champion for ultra-low-power microcontrollers (52 KB, ~36 µs).
   * **1D CNN**: Champion for Software-Defined Radio (SDR) and FPGA baseband processors (0.21 MB, ~424 µs).
   * **Random Forest**: Parameter explosion (140M nodes, 1 GB) restricts deployment to baseband servers.

---

## Presentation Slides

The repository includes publication-grade PowerPoint presentations compliant with the course template:
* 📊 **[Complete 20-Slide Master Presentation](EE136IU_Topic1_Complete_Benchmark_Presentation.pptx)**
* 📈 **[Classical ML Standalone Presentation](EE136IU_Topic1_Classical_ML_Presentation.pptx)**

---

## Authors & Acknowledgments

* **Course**: EE136 — Machine Learning for Wireless Communications
* **Institution**: International University — VNU-HCM (EE136IU)
* **Topic 1**: Comprehensive Model Benchmark (Classical ML vs. MLP vs. 1D CNN)
