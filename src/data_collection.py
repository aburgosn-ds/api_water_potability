import pandas as pd
import os
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/water_potability.csv")

train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

data_path = os.path.join("data", "raw")
os.makedirs(data_path, exist_ok=True)

train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)