import pandas as pd
from scipy.io import savemat

# Load CSV
df = pd.read_csv(r"C:\Users\Shirsha Pattanaik\gtzan_features.csv")

# Separate features & labels
features = df.drop("genre", axis=1).values
labels = df["genre"].values

# Save to MAT file
savemat("gtzan_data.mat", {
    "features": features,
    "labels": labels
})

print("MAT file saved as gtzan_data.mat")
