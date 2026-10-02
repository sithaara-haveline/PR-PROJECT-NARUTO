import pandas as pd

df = pd.read_csv("hand_seal_dataset.csv")

# Look at non-zero features for Left hand (e.g., landmark 9 middle MCP)
l_cols = [c for c in df.columns if c.startswith("L_")]
print("Sample Left hand features from CSV:")
print(df[l_cols].iloc[0].dropna().head(12))