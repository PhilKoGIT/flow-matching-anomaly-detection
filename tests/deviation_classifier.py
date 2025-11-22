
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

train_data, test_data = (df, test_size=0.2, random_state=42)

#welche hyperparameter muss ich in der inference einhalten?



#genau überlegen wie ich das modell einstellen möchte, damit ich die funktione nutzen kann
#n_batch: größe des batches nutzt bei n_batch>0 den iterator. sollte egal sein...