import pandas as pd
from pathlib import Path

CSV_PATH = "./Data/Raw/list_attr_celeba.csv"

df = pd.read_csv(CSV_PATH)

image_name = "000226.jpg"

row = df[df["image_id"] == image_name]

print(row[["image_id", "Eyeglasses", "Wearing_Hat", "Young"]])