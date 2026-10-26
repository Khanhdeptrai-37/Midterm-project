# EE136IU • Topic 1: Comprehensive Model Benchmark
## Báo Cáo Kỹ Thuật & Hướng Dẫn Web Dashboard Trực Quan (Song Ngữ Việt - Anh)

---

### ĐƯỜNG LINK ONLINE CÔNG KHAI (KHÔNG CẦN CHUNG MẠNG)

- Link truy cập trực tiếp từ mọi nơi (4G, 5G, mọi mạng WiFi):
  [https://satisfy-nuke-reggae-distributors.trycloudflare.com](https://satisfy-nuke-reggae-distributors.trycloudflare.com)

*(Bạn có thể copy đường link này gửi qua Zalo, Messenger cho bạn bè, thầy cô mở xem ngay lập tức mà không cần kết nối chung mạng WiFi với bạn).*

---

### 1. Phân Tích Toàn Bộ Dữ Liệu Trong Thư Mục `features_train`

Thư mục [`features_train`](file:///d:/download/model/features_train) chứa toàn bộ bộ dữ liệu chuẩn của đồ án với tổng cộng **96,000 đoạn sóng (bursts)**, mỗi đoạn sóng dài $L=128$ mẫu phức $I/Q$:

- **Phân bổ nghiêm ngặt (Strict 70/15/15 Partitioning):**
  - **`train` (Huấn luyện):** 67,200 mẫu (70%) $\rightarrow$ [`features_train.csv`](file:///d:/download/model/features_train/features_train.csv) & [`iq_train.npz`](file:///d:/download/model/features_train/iq_train.npz)
  - **`val` (Kiểm chứng):** 14,400 mẫu (15%) $\rightarrow$ [`features_val.csv`](file:///d:/download/model/features_train/features_val.csv) & [`iq_val.npz`](file:///d:/download/model/features_train/iq_val.npz)
  - **`test` (Kiểm thử):** 14,400 mẫu (15%) $\rightarrow$ [`features_test.csv`](file:///d:/download/model/features_train/features_test.csv) & [`iq_test.npz`](file:///d:/download/model/features_train/iq_test.npz)
- **8 Loại điều chế (Modulations):** `BPSK`, `QPSK`, `8PSK`, `16PSK`, `32PSK`, `16QAM`, `64QAM`, `256QAM`.
- **15 Mức SNR Calibrated:** Từ $-10\text{ dB}$ đến $+18\text{ dB}$ (bước $2\text{ dB}$).
- **2 Kênh truyền:** `AWGN` (7,200 mẫu test) và `Rayleigh3Tap` (7,200 mẫu test).
- **Tính đồng bộ tuyệt đối 100%:** Dòng thứ $i$ trong file `.csv` khớp 100% với tensor sóng thô ở dòng $i$ trong file `.npz`!

---

### 2. Các Phiên Bản Web Dashboard Sẵn Có

Hệ thống đã chuẩn bị sẵn **3 cách khởi chạy**:

| Tùy chọn | Tệp khởi chạy 1-Click | Phạm vi sử dụng |
| :--- | :--- | :--- |
| **Public Online (Toàn Cầu)** | [`run_public_online.bat`](file:///d:/download/model/run_public_online.bat) | **Không cần chung mạng**, tự động mở đường hầm Cloudflare Tunnel công khai |
| **English Edition (Nội bộ)** | [`run_dashboard_en.bat`](file:///d:/download/model/run_dashboard_en.bat) | Chạy bản tiếng Anh tại `http://localhost:8501` |
| **Tiếng Việt (Nội bộ)** | [`run_dashboard.bat`](file:///d:/download/model/run_dashboard.bat) | Chạy bản tiếng Việt tại `http://localhost:8501` |

---

### 3. Cấu Trúc 5 Tab Chuyên Sâu Trong Dashboard

1. **Live Side-by-Side Inference:**
   - 3 Real-time cards: **Random Forest** (Tier 1), **3-Layer MLP** (Tier 2), **1D CNN** (Tier 3).
   - Metrics: Predicted Label, Confidence Score (%), Correct/Confused status badge, Real-time latency ($\mu\text{s}$).
   - 8-Class probability distribution bar charts with clear ground truth indicator.
   - Constellation scatter plot ($I$ vs $Q$) & Time-domain continuous trajectory waveform ($I[n], Q[n]$).
   - 18 Domain features Z-score standardization profile.
2. **Comprehensive Benchmark Gallery:**
   - Accuracy vs. SNR (-10 dB to +18 dB) with theoretical random guess (12.5%) and feature ceiling (48.25%).
   - Macro-F1 vs. SNR curves.
   - Normalized confusion matrices across 3 operational regimes: Low SNR (-4 dB), Mid SNR (+4 dB), High SNR (+14 dB).
   - Full 15-point calibrated grid data table & Executive Summary.
3. **What-If Signal Synthesizer:**
   - Parameter sliders: $C_{40}, C_{20}$, PAPR, $\gamma_{max}$.
   - Quick presets: Ideal BPSK, Ideal QPSK, High PAPR Dense QAM, Severe Multipath.
   - Real-time simultaneous responses from all 3 models.
4. **Dataset Architecture & Explorer:**
   - Total 96,000 bursts summary (Train 70%, Val 15%, Test 15%).
   - 18 Domain features table with physical significance.
   - Zero Data Leakage verification.
5. **RF Baseband Mechanics & Theory:**
   - RMS energy scaling proof ($r_{norm}[n] = r[n]/E_{rms}$).
   - Mathematical proof of the 48.25% feature ceiling.
   - Engineering trade-offs analysis (Memory footprint, latency, FPGA/SDR feasibility).
