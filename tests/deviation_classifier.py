
# test.py

from ForestDiffusion import ForestDiffusionModel
import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
from preprocessing import prepare_data

dataset_path = Path(__file__).parent / "data" / "EngineFaultDB_Final.csv"



df = pd.read_csv(dataset_path)

train_data, test_data = split_data(df, test_size=0.2, random_state=42)


