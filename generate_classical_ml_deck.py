"""
Script to generate the standalone 6-slide Classical ML presentation:
EE136IU_Topic1_Classical_ML_Presentation.pptx
"""

import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

from presentation_utils import (
    BG_WHITE, CARD_BG, CARD_BORDER, TEXT_MAIN, TEXT_MUTED, TECH_BLUE,
    ACCENT_BLUE, TIER1_GREEN, AMBER_WARN, PURPLE_STAT, FONT_TITLE, FONT_BODY, FONT_CODE,
    add_header, add_footer, add_card, set_shape_flat
)

def build_classical_ml_deck():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.500)
    blank_layout = prs.slide_layouts[6]
    total_slides = 6

    # ==========================================
    # SLIDE 1: TITLE SLIDE
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = add_card(s1, Inches(0.80), Inches(0.80), Inches(11.73), Inches(5.90), fill_color=RGBColor(0x0F, 0x17, 0x2A), border_color=RGBColor(0x1E, 0x29, 0x3B))
    
    # Title box
    tx1 = s1.shapes.add_textbox(Inches(1.40), Inches(1.30), Inches(10.50), Inches(4.80))
    tf1 = tx1.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "EE136 • MACHINE LEARNING FOR WIRELESS COMMUNICATIONS"
    p.font.name = FONT_TITLE
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = TECH_BLUE
    p.space_after = Pt(12)

    p = tf1.add_paragraph()
    p.text = "Topic 1: Comprehensive Model Benchmark"
    p.font.name = FONT_TITLE
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = BG_WHITE
    p.space_after = Pt(6)

    p = tf1.add_paragraph()
    p.text = "Tier 1: Classical Machine Learning (Random Forest vs. RBF-SVM)"
    p.font.name = FONT_TITLE
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = TIER1_GREEN
    p.space_after = Pt(18)

    p = tf1.add_paragraph()
    p.text = "Automatic Modulation Classification (AMC) on 18 Domain Features Across 8 Modulation Schemes"
    p.font.name = FONT_BODY
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
    p.space_after = Pt(28)

    # Pill tags
    tags = [
        ("DATASET", "96,000 Bursts (Train 70% / Val 15% / Test 15%)"),
        ("INPUT", "18 Invariant Physical Features (HOC + Welch PSD + PAPR)"),
        ("MODELS", "Random Forest (200 Trees) & Support Vector Machine (RBF, C=10)"),
        ("CHANNELS", "AWGN & Rayleigh 3-Tap Fading (-10 dB to +18 dB)")
    ]
    for tag_name, tag_val in tags:
        p = tf1.add_paragraph()
        r1 = p.add_run()
        r1.text = f"• {tag_name}: "
        r1.font.bold = True
        r1.font.size = Pt(11)
        r1.font.color.rgb = TECH_BLUE
        r2 = p.add_run()
        r2.text = tag_val
        r2.font.bold = False
        r2.font.size = Pt(11)
        r2.font.color.rgb = RGBColor(0xCB, 0xD5, 0xE1)
        p.space_after = Pt(4)

    # ==========================================
    # SLIDE 2: FEATURE DISTILLATION & ZERO DATA LEAKAGE
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "TOPIC 01 • DOMAIN FEATURE EXTRACTION & PIPELINE", "Feature Distillation: Raw Baseband to 18 Invariant Domain Features")
    add_footer(s2, 2, total_slides, "Topic 1 • Classical Machine Learning")

    # Card 1: Raw I/Q to 18 Features
    c1 = add_card(s2, Inches(0.80), Inches(1.35), Inches(3.65), Inches(5.25))
    tx = s2.shapes.add_textbox(Inches(0.95), Inches(1.50), Inches(3.35), Inches(4.95))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Raw I/Q Dimension Reduction"
    p.font.name = FONT_TITLE
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    p.space_after = Pt(8)

    items1 = [
        ("Input Burst:", "128 complex baseband samples r[n] = I[n] + jQ[n] (256 raw real values)."),
        ("Vulnerability:", "Raw I/Q suffers high variance under random channel phase rotation θ and AWGN."),
        ("14x Compression:", "Condenses 256 dimensions down to 18 robust domain statistical invariants."),
        ("Zero Data Leakage:", "StandardScaler parameters (mean μ, std σ) computed solely on 67,200 Train bursts. Never exposed to Test split.")
    ]
    for pref, text in items1:
        p = tf.add_paragraph()
        r1 = p.add_run(); r1.text = pref + " "; r1.font.bold = True; r1.font.color.rgb = TECH_BLUE; r1.font.size = Pt(11)
        r2 = p.add_run(); r2.text = text; r2.font.color.rgb = TEXT_MAIN; r2.font.size = Pt(11)
        p.space_after = Pt(6)

    # Card 2: 18 Feature Categories
    c2 = add_card(s2, Inches(4.65), Inches(1.35), Inches(4.35), Inches(5.25))
    tx = s2.shapes.add_textbox(Inches(4.80), Inches(1.50), Inches(4.05), Inches(4.95))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "18 Physical Feature Taxonomy"
    p.font.name = FONT_TITLE
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TECH_BLUE
    p.space_after = Pt(8)

    items2 = [
        ("Higher-Order Cumulants (9):", "C20, C21, C40, C41, C42, C60, C61, C62, C63. Insensitive to Gaussian noise; isolates constellation rotational symmetries."),
        ("Phase Invariance (C40):", "BPSK (|C40|=2.0) and QPSK (|C40|=1.0) decouple cleanly. M>=8 PSK converge to |C40|=0."),
        ("Envelope & PAPR (2):", "sigma_aa (std dev of envelope) and PAPR (Peak-to-Average Power Ratio). Separates constant-envelope PSK from multi-amplitude QAM."),
        ("Welch PSD & Phase Stats (7):", "gamma_max, Kurt_psd, Skew_psd, H_psd, H_phase, sigma_ap, sigma_dp measure spectral concentration and angular spread.")
    ]
    for pref, text in items2:
        p = tf.add_paragraph()
        r1 = p.add_run(); r1.text = pref + " "; r1.font.bold = True; r1.font.color.rgb = TECH_BLUE; r1.font.size = Pt(11)
        r2 = p.add_run(); r2.text = text; r2.font.color.rgb = TEXT_MAIN; r2.font.size = Pt(11)
        p.space_after = Pt(6)

    # Card 3: Feature Importance preview / stats
    c3 = add_card(s2, Inches(9.20), Inches(1.35), Inches(3.33), Inches(5.25))
    tx = s2.shapes.add_textbox(Inches(9.35), Inches(1.50), Inches(3.03), Inches(4.95))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Feature Invariance Power"
    p.font.name = FONT_TITLE
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TIER1_GREEN
    p.space_after = Pt(8)

    items3 = [
        ("Energy Normalization:", "All bursts RMS-scaled: r_norm[n] = r[n] / sqrt(1/L sum |r[k]|^2). Prevents trivial power-level cheating."),
        ("Phase Invariance:", "Cumulant magnitudes |C_pq| eliminate unknown phase offset e^(jθ) introduced by Rayleigh multipath."),
        ("Theoretical Ceiling:", "Cumulants up to 6th order cannot resolve 8PSK/16PSK/32PSK internal phase angles. Sets mathematical ceiling ~48.25%.")
    ]
    for pref, text in items3:
        p = tf.add_paragraph()
        r1 = p.add_run(); r1.text = pref + " "; r1.font.bold = True; r1.font.color.rgb = TECH_BLUE; r1.font.size = Pt(11)
        r2 = p.add_run(); r2.text = text; r2.font.color.rgb = TEXT_MAIN; r2.font.size = Pt(11)
        p.space_after = Pt(8)

    # ==========================================
    # SLIDE 3: MODEL ARCHITECTURES & PARAMETER COMPLEXITY
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "TOPIC 01 • ALGORITHM ARCHITECTURES & COMPLEXITY", "Classical ML Specification: Random Forest (Bagging) vs. SVM (RBF)")
    add_footer(s3, 3, total_slides, "Topic 1 • Classical Machine Learning")

    # Left: Random Forest
    c_rf = add_card(s3, Inches(0.80), Inches(1.35), Inches(5.65), Inches(5.25))
    tx = s3.shapes.add_textbox(Inches(0.95), Inches(1.50), Inches(5.35), Inches(4.95))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Tier 1A: Random Forest Classifier (Ensemble Bagging)"
    p.font.name = FONT_TITLE
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = TIER1_GREEN
    p.space_after = Pt(8)

    rf_bullets = [
        ("Ensemble Structure:", "200 decision trees (n_estimators=200), bootstrap sampling on 67,200 training instances."),
        ("Splitting Criterion:", "Gini Impurity with unconstrained depth (max_depth=None). Subsamples sqrt(18) ≈ 4 features at each split node."),
        ("Parameter Explosion:", "Deep trees grow without pruning -> 140,572,728 total decision parameters across 200 trees!"),
        ("Model Checkpoint Size:", "999.7 MB (.pkl file size). Severe RAM footprint when loaded into memory."),
        ("Training Complexity:", "Parallelized multi-core training (n_jobs=-1) completes in 121.03 s (~2.0 min on CPU)."),
        ("Inference Speed:", "Fast execution: ~280 µs single-sample, 177 µs batched. Well within 5G frame budget (1.0 ms).")
    ]
    for pref, text in rf_bullets:
        p = tf.add_paragraph()
        r1 = p.add_run(); r1.text = pref + " "; r1.font.bold = True; r1.font.color.rgb = TECH_BLUE; r1.font.size = Pt(11)
        r2 = p.add_run(); r2.text = text; r2.font.color.rgb = TEXT_MAIN; r2.font.size = Pt(11)
        p.space_after = Pt(4)

    # Right: SVM (RBF)
    c_svm = add_card(s3, Inches(6.88), Inches(1.35), Inches(5.65), Inches(5.25))
    tx = s3.shapes.add_textbox(Inches(7.03), Inches(1.50), Inches(5.35), Inches(4.95))
    tf = tx.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Tier 1B: Support Vector Machine (RBF Kernel)"
    p.font.name = FONT_TITLE
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    p.space_after = Pt(8)

    svm_bullets = [
        ("Mathematical Formulation:", "Dual QP optimization finding maximum-margin separating hyperplanes in infinite-dimensional RKHS space."),
        ("Hyperparameters:", "Radial Basis Function (RBF) kernel: K(x, x') = exp(-gamma * ||x-x'||^2), Regularization C=10, gamma='scale' = 1/(18 * Var(X))."),
        ("Support Vector Count:", "1,035,568 support vector coefficients (57,531 support vectors across 8 classes * 18 features)."),
        ("Model Checkpoint Size:", "14.96 MB (.pkl file size). 66x more compact than Random Forest on disk."),
        ("Training Bottleneck:", "Quadratic O(N^2) optimization on 67,200 bursts requires 1,788.23 s (~29.8 min CPU) — 15x slower to train!"),
        ("Inference Latency Trap:", "Must compute kernel dot products against 57k vectors per sample -> ~9,700 µs (9.7 ms). Fails real-time 5G deadline (1.0 ms)!")
    ]
    for pref, text in svm_bullets:
        p = tf.add_paragraph()
        r1 = p.add_run(); r1.text = pref + " "; r1.font.bold = True; r1.font.color.rgb = TECH_BLUE; r1.font.size = Pt(11)
        r2 = p.add_run(); r2.text = text; r2.font.color.rgb = TEXT_MAIN; r2.font.size = Pt(11)
        p.space_after = Pt(4)

    # ==========================================
    # SLIDE 4: EMPIRICAL TRAINING PROFILING & TRADE-OFFS
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "TOPIC 01 • EMPIRICAL PROFILING & SYSTEM TRADE-OFFS", "Classical ML System Profiling: Compute, Memory & Latency Dilemma")
    add_footer(s4, 4, total_slides, "Topic 1 • Classical Machine Learning")

    # Table of comparison
    t_card = add_card(s4, Inches(0.80), Inches(1.35), Inches(11.73), Inches(2.70))
    # Add table inside
    table_shape = s4.shapes.add_table(5, 5, Inches(1.00), Inches(1.50), Inches(11.33), Inches(2.35))
    table = table_shape.table
    table.columns[0].width = Inches(2.60)
    table.columns[1].width = Inches(2.18)
    table.columns[2].width = Inches(2.18)
    table.columns[3].width = Inches(2.18)
    table.columns[4].width = Inches(2.18)

    headers = ["Metric / Specification", "Random Forest (RF)", "SVM (RBF Kernel)", "Ratio (RF vs. SVM)", "Engineering Impact"]
    rows = [
        ["Training Duration (CPU)", "121.03 s (~2.0 min)", "1,788.23 s (~29.8 min)", "SVM is 14.8x slower", "RF enables rapid retraining on SDR nodes"],
        ["Trainable Parameters", "140,572,728 tree nodes", "1,035,568 SV weights", "RF has 135x more params", "RF suffers severe parameter explosion"],
        ["Checkpoint File Size", "999.7 MB (0.98 GB)", "14.96 MB (0.015 GB)", "RF is 66x larger on disk", "SVM fits on flash; RF requires large DRAM"],
        ["Single-Burst Latency (B=1)", "~280 µs (0.28 ms)", "~9,700 µs (9.70 ms)", "RF is 35x faster in infer.", "RF satisfies 5G 1ms budget; SVM FAILS"]
    ]
    for c_idx, h in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.text = h
        cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        p = cell.text_frame.paragraphs[0]
        p.font.name = FONT_TITLE; p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = BG_WHITE
        p.alignment = PP_ALIGN.CENTER

    for r_idx, r_data in enumerate(rows):
        for c_idx, val in enumerate(r_data):
            cell = table.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if r_idx % 2 == 0 else RGBColor(0xF8, 0xFA, 0xFC)
            p = cell.text_frame.paragraphs[0]
            p.font.name = FONT_BODY; p.font.size = Pt(9.5); p.font.color.rgb = TEXT_MAIN
            if c_idx == 0:
                p.font.bold = True
            elif c_idx == 3:
                p.font.bold = True; p.font.color.rgb = TECH_BLUE
            elif c_idx == 4:
                p.font.color.rgb = AMBER_WARN if "FAILS" in val else TIER1_GREEN

    # 2 Bottom Cards explaining Trade-offs
    c_b1 = add_card(s4, Inches(0.80), Inches(4.25), Inches(5.65), Inches(2.35))
    tx = s4.shapes.add_textbox(Inches(0.95), Inches(4.35), Inches(5.35), Inches(2.15))
    tf = tx.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Why We Drop SVM & Keep Random Forest in Production"
    p.font.name = FONT_TITLE; p.font.size = Pt(12.5); p.font.bold = True; p.font.color.rgb = AMBER_WARN
    p.space_after = Pt(4)
    bullets_b1 = [
        ("Latency Violation:", "SVM latency (~9.7 ms) exceeds the 3GPP 5G NR TTI frame budget (1.0 ms) by 9.7x! Unusable for real-time SDR baseband."),
        ("Support Vector Drag:", "Evaluating 57,531 dual coefficients per burst creates severe computation bottlenecks without accuracy gain (SVM 33.8% vs RF 33.8%).")
    ]
    for pref, text in bullets_b1:
        p = tf.add_paragraph()
        r1 = p.add_run(); r1.text = pref + " "; r1.font.bold = True; r1.font.color.rgb = TECH_BLUE; r1.font.size = Pt(10.5)
        r2 = p.add_run(); r2.text = text; r2.font.color.rgb = TEXT_MAIN; r2.font.size = Pt(10.5)
        p.space_after = Pt(2)

    c_b2 = add_card(s4, Inches(6.88), Inches(4.25), Inches(5.65), Inches(2.35))
    tx = s4.shapes.add_textbox(Inches(7.03), Inches(4.35), Inches(5.35), Inches(2.15))
    tf = tx.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Random Forest: High Memory, But Fast Decision Speed"
    p.font.name = FONT_TITLE; p.font.size = Pt(12.5); p.font.bold = True; p.font.color.rgb = TIER1_GREEN
    p.space_after = Pt(4)
    bullets_b2 = [
        ("Instant Logic Trees:", "RF executes sequential threshold comparisons: if C40 <= -1.45 goto Left else goto Right. No matrix inversions!"),
        ("Live Dashboard Integration:", "At ~280 µs latency, RF responds instantaneously on user clicks in the live demo deliverable.")
    ]
    for pref, text in bullets_b2:
        p = tf.add_paragraph()
        r1 = p.add_run(); r1.text = pref + " "; r1.font.bold = True; r1.font.color.rgb = TECH_BLUE; r1.font.size = Pt(10.5)
        r2 = p.add_run(); r2.text = text; r2.font.color.rgb = TEXT_MAIN; r2.font.size = Pt(10.5)
        p.space_after = Pt(2)

    # ==========================================
    # SLIDE 5: PERFORMANCE EVALUATION ACROSS 15 SNRS
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "TOPIC 01 • STANDARDIZED BENCHMARK EVALUATION", "Classical ML Performance: Accuracy & Macro-F1 Across 15 Calibrated SNRs")
    add_footer(s5, 5, total_slides, "Topic 1 • Classical Machine Learning")

    # Image Left
    img_path = os.path.join('extracted_media', 'ml_metrics_snr.png')
    if os.path.exists(img_path):
        s5.shapes.add_picture(img_path, Inches(0.80), Inches(1.35), Inches(7.50), Inches(5.25))

    # Right Card: Analysis
    c_snr = add_card(s5, Inches(8.50), Inches(1.35), Inches(4.03), Inches(5.25))
    tx = s5.shapes.add_textbox(Inches(8.65), Inches(1.50), Inches(3.73), Inches(4.95))
    tf = tx.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "SNR Regime Breakdown"
    p.font.name = FONT_TITLE; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = TECH_BLUE
    p.space_after = Pt(8)

    snr_breakdown = [
        ("Low SNR (-10 to -2 dB):", "Acc: 14.9% -> 26.5%. Dominated by AWGN noise floor. High noise variances heavily perturb high-order cumulants."),
        ("Steep Transition (0 to +6 dB):", "Acc jumps from 27.6% to 40.4%. BPSK and QPSK emerge with high precision. Signal power overcomes noise floor."),
        ("High-SNR Ceiling (+8 to +18 dB):", "Acc flattens near 48-51% (RF: 50.10%, SVM: 51.35%). Intra-group ambiguity between higher PSK and QAM sets a mathematical wall."),
        ("Macro-F1 Alignment:", "Macro-F1 tracks Accuracy within 0.5% across all 15 SNRs, confirming zero class bias on balanced test splits.")
    ]
    for pref, text in snr_breakdown:
        p = tf.add_paragraph()
        r1 = p.add_run(); r1.text = pref + " "; r1.font.bold = True; r1.font.color.rgb = TECH_BLUE; r1.font.size = Pt(11)
        r2 = p.add_run(); r2.text = text; r2.font.color.rgb = TEXT_MAIN; r2.font.size = Pt(11)
        p.space_after = Pt(6)

    # ==========================================
    # SLIDE 6: ERROR DIAGNOSTICS & CONFUSION MATRICES
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "TOPIC 01 • ERROR DIAGNOSTICS & FEATURE IMPORTANCE", "Classical ML Error Diagnostics: Confusion Matrices & Feature Ranking")
    add_footer(s6, 6, total_slides, "Topic 1 • Classical Machine Learning")

    # Image Left: Confusion matrices
    img_cm = os.path.join('extracted_media', 'ml_confusion_matrices.png')
    if os.path.exists(img_cm):
        s6.shapes.add_picture(img_cm, Inches(0.80), Inches(1.35), Inches(8.10), Inches(5.25))

    # Right Card: Diagnostics & Feature importance
    c_diag = add_card(s6, Inches(9.10), Inches(1.35), Inches(3.43), Inches(5.25))
    tx = s6.shapes.add_textbox(Inches(9.25), Inches(1.50), Inches(3.13), Inches(4.95))
    tf = tx.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Key Diagnostic Findings"
    p.font.name = FONT_TITLE; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = ACCENT_BLUE
    p.space_after = Pt(6)

    diag_items = [
        ("Low SNR (-4 dB | ~22%):", "Severe off-diagonal noise scattering. Cumulant estimations fail; models guess almost uniformly across modulation families."),
        ("Mid SNR (+4 dB | ~35%):", "BPSK is recognized almost perfectly (~98%) via C40=-2.0. QPSK separates cleanly (~57%). Higher-order PSK/QAM remain heavily blurred."),
        ("High SNR (+14 dB | ~47%):", "Mathematical ceiling ceiling: 8PSK, 16PSK, and 32PSK cluster together at |C40|=0. Similarly, 16QAM, 64QAM, and 256QAM confuse each other."),
        ("Top Gini Features:", "1. C40 (BPSK/QPSK divider)\n2. C42 (energy variance)\n3. PAPR (PSK vs QAM)\n4. sigma_aa (envelope std)\n5. H_phase (phase entropy)")
    ]
    for pref, text in diag_items:
        p = tf.add_paragraph()
        r1 = p.add_run(); r1.text = pref + " "; r1.font.bold = True; r1.font.color.rgb = TECH_BLUE; r1.font.size = Pt(10)
        r2 = p.add_run(); r2.text = text; r2.font.color.rgb = TEXT_MAIN; r2.font.size = Pt(10)
        p.space_after = Pt(4)

    out_file = 'EE136IU_Topic1_Classical_ML_Presentation.pptx'
    prs.save(out_file)
    print(f'Successfully built {out_file} with {len(prs.slides)} slides!')

if __name__ == '__main__':
    build_classical_ml_deck()
