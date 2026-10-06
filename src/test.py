import pandas as pd
import joblib
from train import feature_engineering

# Select data
# Read dataframe

X = pd.read_csv("../data/raw/test.csv", index_col = 'id')
X = feature_engineering(X)

model = joblib.load("../models/model.pkl")
predictions = model.predict(X)

df_result = pd.DataFrame({
    'Will_Buy_EV': predictions
}, index = X.index)
df_result.to_csv("../submissions/submission.csv")

print('Done!')