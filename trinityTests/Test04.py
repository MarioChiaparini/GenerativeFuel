import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"")
df = df.sort_values("Energy", ascending=False)
top_200 = df["SMILES"][:200]
model = ChemProp(
    n_layers=3,
    hidden_size=128,
    dropout=0.1,
    loss="mse",
    metrics=["mae", "rmse"],
)
