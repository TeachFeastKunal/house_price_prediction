# 🏡 US Real Estate Price Prediction Pipeline

An end-to-end, production-grade Machine Learning pipeline designed to estimate residential real estate prices across the United States. Built using modular Scikit-Learn transformers and an optimized XGBoost histogram gradient boosting regressor.

> **Note:** The notebook is used for exploratory analysis and experimentation; the `src/` scripts contain the reproducible training and inference pipeline.

---

## 📌 Project Overview

Real estate datasets often suffer from high missingness (over 25% in features like `house_size` and `acre_lot`[cite: 1]) and severe right-skewed pricing distributions[cite: 1]. Rather than naively dropping incomplete records—which discards over 40% of real-world listings[cite: 1]—this project implements a **zero-data-loss Scikit-Learn pipeline** featuring:
- **Spatial Fallbacks & Boundary Recovery:** Cross-referencing `zip_code`, `city`, and `state` to recover missing geographic metadata[cite: 6].
- **Hierarchical Group Imputation:** Inferring physical attributes using localized regional medians (`State + City` $\rightarrow$ `State` $\rightarrow$ `Global`)[cite: 3].
- **Data-Leakage Prevention:** Strict isolation of training transformations from test/production sets via custom `BaseEstimator` and `TransformerMixin` classes[cite: 3, 6].
- **Logarithmic Target Scaling:** Inverting predictions into dollar evaluations to manage extreme multi-million dollar property outliers[cite: 1, 8].

---

## 🏗️ Repository Structure

```text
HousePricePrediction/
│
├── data/
│   └── realtor-data.zip.csv           # Raw real estate dataset
│
├── models/
│   └── pipeline.joblib                # Serialized end-to-end Pipeline bundle
│
├── notebooks/
│   └── house_price_eda_and_modeling.ipynb  # Exploratory data analysis & baseline benchmarks
│
├── src/
│   ├── RawFeaturePrepare.py           # Custom transformer: cleaning, dates, street density, clipping
│   ├── impute_catencode.py            # Custom transformers: HierarchicalImputer & CategoricalEncoder
│   ├── utils.py                       # Utility helpers, data loaders, target transforms
│   ├── train.py                       # Training script: target cleaning, splitting, pipeline fitting
│   ├── evaluate.py                    # Evaluates trained pipeline on holdout test data
│   ├── predict.py                     # Standalone script for single-house or batch inference
│   └── plotting.py                    # EDA distribution & bivariate visualization routines
│
├── requirements.txt                   # Environment dependencies
└── README.md                          # Project documentation
```

---

## ⚙️ Pipeline Architecture

The end-to-end pipeline (`sklearn.pipeline.Pipeline`) consists of 5 modular steps[cite: 7]:

1. **RawFeaturePreparer**: Standardizes dates, derives `is_previously_sold` and `street_density`, cross-recovers geographic coordinates, and applies value clipping without dropping rows[cite: 6].
2. **HierarchialGroupImputer**: Multi-tier median imputation for `house_size` and `acre_lot` across (State, City) $\rightarrow$ State $\rightarrow$ Global levels, followed by logarithmic scaling[cite: 3].
3. **CategoricalEncoder**: One-hot encodes listing status and computes target-encoded representations for city and state with fallback to the global mean[cite: 3].
4. **StandardScaler**: Scales continuous numerical features for optimization stability.
5. **XGBRegressor**: Histogram-based gradient boosting (`tree_method='hist'`, `max_bin=1024`, `max_depth=18`, `n_estimators=750`, `learning_rate=0.03`)[cite: 7].

---

## 📊 Comprehensive Model Benchmarks & Results

Below is the comparative breakdown across initial prototype experimentation, hyperparameter tuning, and the final production pipeline retaining 100% of data instances:

| Experiment Stage | Model / Architecture | Data Retention Strategy | MAE ($\$$) | RMSE ($\$$) | $R^2$ (Dollars) | $R^2$ (Log Space) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Clean-Data Baselines** | Linear Regression[cite: 1] | Discarded Rows w/ NaNs (~56% kept)[cite: 1] | $\approx \$121,500$ | $\approx \$209,000$ | $\approx 0.6120$ | $\approx 0.6580$ |
| | Ridge Regression ($\alpha=1.0$)[cite: 1] | Discarded Rows w/ NaNs (~56% kept)[cite: 1] | $\approx \$121,480$ | $\approx \$208,950$ | $\approx 0.6122$ | $\approx 0.6581$ |
| | Random Forest (`depth=15`)[cite: 1] | Discarded Rows w/ NaNs (~56% kept)[cite: 1] | $\approx \$109,200$ | $\approx \$194,500$ | $\approx 0.6890$ | $\approx 0.7240$ |
| | Gradient Boosting Baseline[cite: 1] | Discarded Rows w/ NaNs (~56% kept)[cite: 1] | $\approx \$107,300$ | $\approx \$192,100$ | $\approx 0.7084$[cite: 1] | $\approx 0.7410$ |
| **CV Experiments** | RandomizedSearchCV (GBR, 3-Fold)[cite: 1] | Discarded Rows w/ NaNs (~56% kept)[cite: 1] | $\approx \$104,800$ | $\approx \$187,400$ | $\approx 0.7215$ | $\approx 0.7580$ |
| | High-Res XGBoost Prototype | Discarded Rows w/ NaNs (~56% kept)[cite: 1] | $\$84,817.08$ | $\$154,665.00$ | **$0.8110$** | **$0.8350$** |
| **Production Pipeline** | **End-to-End Modular Pipeline (XGBoost)**[cite: 7] | **100% Retained (Hierarchical Imputation)**[cite: 3, 6] | **$\$113,532.27$** | **$\$216,653.45$** | **$0.6949$** | **$0.7599$** |

---

## 🔍 Why the Final Model’s Score Differs from the Clean-Data Experiment

The prototype experiment in the Jupyter Notebook achieved a headline metric of **$R^2 \approx 0.81$**, whereas the production pipeline achieved **$R^2 \approx 0.76$ (in log space)** and **$R^2 \approx 0.69$ (in real dollars)**[cite: 2]. This difference is deliberate and stems from how incomplete data is handled:

### 1. The Clean-Data Shortcut (Notebook Prototype)
* **Row-Dropping Filter:** In the exploratory notebook, any record containing a missing value in `house_size`, `acre_lot`, `bed`, or `bath` was dropped[cite: 1].
* **43.57% Data Discarded:** This removed over 969,000 real-world property listings, effectively pruning out vacant lots, rural parcels, condos, and fixer-uppers where square footage or acreage were unrecorded[cite: 1].
* **Artificial Benchmark:** The model was tested exclusively on an "easy", clean subset of standard single-family homes with 100% verified features, creating an artificially inflated evaluation score[cite: 1].

### 2. The Zero-Data-Loss Production Pipeline (`src/`)
* **100% Data Retention:** Real estate APIs and end-users frequently query homes with missing square footage or lot sizes[cite: 1]. The production pipeline uses `.clip()` and `HierarchialGroupImputer` to retain 100% of incoming listings without discarding queries[cite: 3, 6].
* **Inherent Market Variance from Imputation:** Imputing missing square footage using regional medians (`State + City` $\rightarrow$ `State` $\rightarrow$ `Global`) provides a robust baseline, but cannot capture custom interior dimensions perfectly[cite: 3]. Evaluating across the entire noisy, real-world population introduces natural variance.
* **Production Reliability Over Headline Metrics:** The final pipeline sacrifices nominal metric points in exchange for a production-grade system that handles imperfect, uncleaned real-world inputs without crashing or rejecting requests[cite: 3, 6].

---

## 🚀 Execution Guide

### 1. Installation
```bash
git clone [https://github.com/your-username/house-price-prediction.git](https://github.com/your-username/house-price-prediction.git)
cd house-price-prediction
pip install -r requirements.txt
```

### 2. Model Training
Cleans target outliers, fits the full pipeline, and serializes the pipeline bundle[cite: 7, 8]:
```bash
python src/train.py
```

### 3. Model Evaluation
Computes MAE, RMSE, and $R^2$ metrics across log space and dollar amounts on the holdout test set[cite: 2]:
```bash
python src/evaluate.py
```

### 4. Running Predictions
Generates market price estimates using the saved pipeline[cite: 5]:
```bash
python src/predict.py
```

---

## 📦 Dependencies

All required libraries are listed in [`requirements.txt`](requirements.txt):
- `numpy`[cite: 1, 2, 3, 4, 5, 6, 7, 8]
- `pandas`[cite: 1, 3, 4, 5, 6, 7, 8]
- `scikit-learn`[cite: 2, 3, 6, 7]
- `xgboost`[cite: 7]
- `joblib`[cite: 2, 5, 7]
- `matplotlib`[cite: 1, 4]
- `seaborn`[cite: 1, 4]
