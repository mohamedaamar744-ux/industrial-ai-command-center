# 🛡️ MetroGuard AI — Industrial AI Command Center

**AI-Powered Predictive Maintenance for Rail Compressor Systems**

*Smarter Maintenance. Safer Journeys.*

An end-to-end predictive maintenance system built entirely on **real industrial sensor data** from a metro train's Air Production Unit (compressor).

This project covers the complete Data Science lifecycle — from raw sensor data to machine learning, anomaly detection, clustering, explainability, and a production-style monitoring dashboard — following the same rigor a professional ML team would apply: verified assumptions, honest limitations, documented experiments, and metrics chosen for the problem rather than appearance.

---

## 🎯 Problem Statement

Unplanned compressor failure on a metro train can cause service disruption, costly emergency repairs, and safety risk.

This project asks:

> **Can we detect and understand compressor failure risk from sensor telemetry alone — reliably, explainably, and honestly about what the data can and cannot support?**

---

## 🗃️ Dataset

[**MetroPT-3**](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset) (UCI Machine Learning Repository) — real multivariate time-series data from a compressor's Air Production Unit on an operational Porto metro train.

| Property                 | Details                                                         |
| ------------------------ | --------------------------------------------------------------- |
| Sensors                  | 7 analog (pressure, oil temperature, motor current) + 8 digital |
| Period                   | February – August 2020                                          |
| Readings                 | ~1.5M                                                           |
| Sampling                 | ~10-second intervals — verified from the provided file          |
| Confirmed failures       | 4 documented events (air leak)                                  |
| Failure-labeled readings | ~2%                                                             |

---

## 🧭 Methodology

The project follows a CRISP-DM-inspired workflow:

```text
Data Understanding
        ↓
EDA
        ↓
Feature Engineering
        ↓
Classification
        ↓
Anomaly Detection
        ↓
Clustering & PCA
        ↓
Explainability
        ↓
Dashboard
```

Each major stage is documented in its own notebook, including experiments that were tested but not adopted.

---

## 🔍 Key Findings

* **Sampling rate mismatch:** the provided file contains approximately 10-second intervals rather than the documented 1 Hz sampling rate. This was discovered through timestamp verification.
* **Failure signature:** TP2 pressure pulses become faster with a lower peak after failure onset; H1/TP3 become inactive; oil temperature rises and remains elevated; motor current shifts from cyclic to more continuous behavior.
* **Detection delay:** approximately 1–2 minutes of abnormal behavior are required before rolling features begin to diverge from normal.
* **Class imbalance:** failure-labeled observations represent only a small fraction of the dataset, so Precision, Recall, F1, and PR-AUC are emphasized instead of accuracy.

---

## 🤖 Models & Results

| Task              | Model                                | Result                                                   | Notes                                        |
| ----------------- | ------------------------------------ | -------------------------------------------------------- | -------------------------------------------- |
| Classification    | **Logistic Regression** (scaled)     | P=0.62, R=0.92, **F1=0.74**                              | Final supervised model                       |
| Anomaly Detection | **One-Class SVM** (normal data only) | P=0.75, R=0.68, **F1=0.72–0.76**                         | Failure labels were not used during training |
| Clustering        | **K-Means (K=4)**                    | "Full Load" cluster → **43.9% failure-labeled readings** | Reflected the EDA failure signature          |
| Explainability    | **SHAP (LinearExplainer)**           | TP2 & oil temperature among strongest predictors         | Supports EDA findings                        |

Random Forest and XGBoost were tested across multiple configurations, including class weighting, threshold tuning, and tree-depth variations. They consistently underperformed Logistic Regression under the tested configurations.

Detailed experiments are documented in `04_classification_baseline.ipynb` and `experiments/`.

---

## 🧠 Design Decisions

| Decision                                        | Reasoning                                                                      |
| ----------------------------------------------- | ------------------------------------------------------------------------------ |
| Chronological train/test split                  | Prevents time-series data leakage — the test period was held out entirely      |
| 1-minute aggregation within continuity segments | Prevents averaging across timestamp gaps and improved Anomaly Detection        |
| Failure #4 excluded from pre-failure analysis   | A 14h11m gap precedes it, making pre-failure analysis unreliable               |
| Failure #4 retained in target labels            | The documented failure period itself contains usable observations              |
| No RUL on MetroPT-3                             | Only 3–4 independent failure events — too few for a reliable regression target |
| `diff` / lag features tested separately         | No measurable benefit in the clean A/B experiment                              |

Full trade-off log: `NOTES.md`

---

## 📊 Dashboard — MetroGuard AI

A dark-themed, multi-page Streamlit application built on the trained models — not a static report.

| Page              | Purpose                        |
| ----------------- | ------------------------------ |
| Overview          | Current system health          |
| Live Monitoring   | Current machine behavior       |
| Anomaly Detection | Abnormal operating periods     |
| Machine States    | Discovered operating modes     |
| Model Analysis    | Model performance              |
| Explainability    | Why an observation was flagged |
| Data Explorer     | Underlying sensor data         |

### Run Locally

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

🔗 **Live Dashboard:** 

---

## 📁 Project Structure

```text
industrial-ai-command-center/
│
├── README.md
├── NOTES.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_classification_baseline.ipynb
│   ├── 05_anomaly_detection.ipynb
│   ├── 06_clustering.ipynb
│   └── 07_explainability.ipynb
│
├── experiments/
│   └── ab_test_diff.ipynb
│
├── dashboard/
│   ├── app.py
│   ├── shared.py
│   ├── pages/
│   └── requirements.txt
│
├── rul_case_study/
│   └── NASA CMAPSS
│
└── assets/
    └── images/
```

---

## ⚠️ Limitations & Honesty Notes

* Only **4 documented failure events** are available, all associated with the same reported failure type.
* Results should not be assumed to generalize to unseen failure modes.
* The One-Class SVM was evaluated across independent training samples, but the dataset still contains a limited number of independent failure events.
* The project does not claim production-level reliability from this dataset alone.
* RUL prediction was deliberately **not** attempted on MetroPT-3 because the available failure history is insufficient for a reliable regression target.

---

## 🔭 Future Work

Potential production-oriented improvements include:

* Root-cause analysis of false positives
* Human-in-the-loop feedback
* Model ensembling
* Confidence-tiered alerts
* Periodic model retraining
* Additional failure modes and operational data

A separate **RUL case study using NASA CMAPSS** is included under `rul_case_study/`.

---

## 📚 Acknowledgments

Veloso, B., Ribeiro, R.P., Pereira, P.M., Gama, J.
*"The MetroPT dataset for predictive maintenance."*
*Scientific Data 9, 764 (2022).*
