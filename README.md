# JGB-Yield-Curve-PCA-Analysis
Principal Component Analysis (PCA) applied to the Japanese Government Bond (JGB) yield curve to extract Level, Slope, and Curvature factors.

---

## **Overview**

This project applies Principal Component Analysis (PCA) to the historical Japanese Government Bond (JGB) yield curve (1Y–40Y) to extract the three dominant factors that explain nearly all movements in the yield curve:

* **Level** – overall interest rate level
* **Slope** – long–short term rate differential
* **Curvature** – relative movement of intermediate maturities

These factors represent the standard decomposition widely used in fixed-income portfolio management, risk modeling, and macroeconomic analysis.

---

##  **Key Results**

### **Variance Explained by Each Factor**

| Factor        | Variance Explained |
| ------------- | ------------------ |
| **Level**     | 96.6%              |
| **Slope**     | 2.3%               |
| **Curvature** | 0.9%               |

Together, the first three principal components explain **~99.8%** of total variation in the JGB yield curve.

### **Interpretation**

* **Level** shifts correspond to macro regime changes and BOJ policy stance
* **Slope** reflects market expectations for recession/expansion and rate cuts
* **Curvature** often reacts to QE/QT operations and supply–demand distortions in mid-term bonds

The factor time series clearly highlight the negative rate environment and BOJ’s 2024 policy normalization.

---

##  **Method**

1. Load JGB yield curve data (CSV)
2. Clean and transform data (convert '-' to NaN, handle types)
3. Extract maturities (1Y–40Y)
4. Apply PCA (3 components)
5. Visualize Level / Slope / Curvature over time
6. Evaluate factor contributions and discuss implications

---

##  **Code Snippet (Minimal Working Example)**

```python
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load
df = pd.read_csv("jgbcm_all.csv", encoding="cp932")

# Use first row as header
header = df.iloc[0]
data = df.iloc[1:].copy()
data.columns = header

# Index by date
data = data.rename(columns={"基準日": "date"})
data.set_index("date", inplace=True)

# Clean data
data = data.replace("-", pd.NA)
num_df = data.astype("Float64").dropna()

# PCA
pca = PCA(n_components=3)
components = pca.fit_transform(num_df.to_numpy())

pc_df = pd.DataFrame(
    components,
    columns=["Level", "Slope", "Curvature"],
    index=num_df.index
)

# Plot
pc_df.plot(figsize=(12, 6))
plt.title("JGB Yield Curve PCA")
plt.grid(True)
plt.show()
```

---

##  **Project Structure**

```
JGB-YieldCurve-PCA/
├── src/
│   └── pca_yieldcurve.py
├── notebooks/
│   └── JGB_PCA_Analysis.ipynb
└── README.md
```

*The dataset is not included due to licensing.

---

##  **Motivation**

The goal of this project is to understand fixed-income factor structures and interest rate dynamics through quantitative methods.
PCA-based factor extraction is fundamental for:

* Bond portfolio construction
* Duration/curve risk management
* Macro regime detection
* Yield curve forecasting models

This analysis provides a foundation for further extensions such as:

* Dynamic PCA
* Nelson–Siegel–Svensson factor extraction
* Macro regressions on Level/Slope/Curvature
* Event studies around BOJ meetings

---

##  **Author**

**Haruka Masatsugu**
Economics Student | Quantitative Finance & Fixed-Income Modeling
Interested in statistics, probability theory, and macro–quant methods.

---

