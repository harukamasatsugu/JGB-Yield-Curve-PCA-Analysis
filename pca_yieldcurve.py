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
