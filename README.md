# 📊 Marketing Mix Modeling (MMM) Case Study

This repository contains a complete MMM workflow using both **OLS** and **Ridge Regression** models.  
The objective is to analyze media spends, derive channel contributions, compute ROI, and simulate budget reallocation.

---

## 📂 Project Structure

```
Pub_mmm_Project/
│
├── data/
│   └── Case_Study_Input_File.xlsx   # Input dataset
│
├── outputs/                         # Model outputs (auto-generated)
│
├── scripts/
│   ├── OLS_Model.py                 # Run OLS-based MMM pipeline
│   └── Ridge_robustness_pipeline.py # Run Ridge regression robustness pipeline
│
├── src/                             # Helper modules
│   ├── eda.py
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── attribution.py
│   ├── diagnostics.py
│   └── utils.py
│
├── EDA.ipynb                        # Notebook with exploratory data analysis
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

---

## ⚙️ Setup

1. Clone or copy the repository.  
2. Create a virtual environment (recommended: conda).  

```bash
conda create -n mmm_env python=3.9 -y
conda activate mmm_env
```

3. Install required packages:  

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Pipelines

### 1. **OLS Model**

Runs an OLS-based MMM pipeline with diagnostics, ROI analysis, and budget simulation.

```bash
python scripts/OLS_Model.py
```

- Outputs saved in `outputs/`
  - `contrib.csv` → Channel contributions
  - `roi.csv` → ROI per channel
  - `spend_map.csv` → Total spend per channel
  - `reallocation_simulation.json` → Budget reallocation impact
  - `diagnostics/` → Rolling coefficients, suspicious ROIs, diagnostics summary

---

### 2. **Ridge Robustness Pipeline**

Runs Ridge regression with **alpha tuning (GridSearchCV + TimeSeriesSplit)** for stability.

```bash
python scripts/Ridge_robustness_pipeline.py
```

- Outputs saved in `outputs/robustness/`
  - `vif.csv` → Collinearity diagnostics
  - `ridge_alpha.json` → Tuned alpha
  - `ridge_coefs_tuned.csv` → Final coefficients
  - `roi_clean.csv` → ROI per channel
  - `reallocation_simulation.json` → Budget reallocation uplift

---

### 3. **EDA Notebook**

Open the notebook for data exploration, distributions, trends, and correlations:

```bash
jupyter notebook EDA.ipynb
```

⚠️ Note: Paths are dynamic (`Path.cwd()`) so no changes are required when moving the project.

---

## 📑 Key Features

- **Adstock Transformation**: Models media carry-over effect.
- **OLS Model**: Interpretable coefficients, but prone to instability with multicollinearity.
- **Ridge Regression**: Adds regularization, tunes alpha for stability.
- **Diagnostics**:
  - Train/Test split evaluation
  - Rolling window coefficient stability
  - ROI sanity check (flags suspicious values)
- **Budget Simulation**:
  - Simulates reallocation (5% and 10%) from low-ROI to high-ROI channels.
  - Outputs expected KPI uplift.

---

## 📊 Example Results

- **OLS Model**  
  - Train R² ≈ 0.57  
  - Test R² often unstable due to small dataset  
  - ROIs sometimes extreme (indicative of collinearity)

- **Ridge Model**  
  - In-sample R² ≈ 0.79  
  - Cross-validation R² negative → weak generalization  
  - More stable coefficients, better for directional insights

---

## ⚠️ Caveats

- This MMM is **observational**, not causal → insights should be validated with experiments or holdouts.  
- Dataset is small (48 rows), which reduces generalizability.  
- High ROI outliers should be carefully reviewed before making budget shifts.  
- Use recommendations as **directional guidance**.

---

## 📌 Recommendations

- Reallocate budget **from TV / Display to WhatsApp & Hotstar** for higher ROI uplift.  
- Validate via **incremental experiments** before scaling.  
- Consider testing non-linear models (e.g., saturation curves, log-log models).  
- Expand dataset for more robust MMM insights.

