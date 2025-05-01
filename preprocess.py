import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

fires = pd.read_csv("sanbul2district-divby100.csv")
fires["burned_area"] = np.log1p(fires["burned_area"])

X = fires.drop("burned_area", axis=1)
y = fires["burned_area"]

cat_attribs = ["month", "day"]
num_attribs = [col for col in X.columns if col not in cat_attribs]

num_pipeline = Pipeline([("scaler", StandardScaler())])
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

X_prepared = full_pipeline.fit_transform(X)

# 저장 경로
os.makedirs("model", exist_ok=True)
joblib.dump(full_pipeline, "model/preprocessor.pkl")
np.save("model/X_train.npy", X_prepared)
np.save("model/y_train.npy", y.values)
