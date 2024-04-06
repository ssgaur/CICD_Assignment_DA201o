import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

model = LogisticRegression(
    # penalty='l2',          # Regularization type (l2: Ridge regularization)
    C=0.01,                 # Inverse of regularization strength (lower values specify stronger regularization)
    # solver='lbfgs',        # Algorithm to use in the optimization problem (lbfgs: Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm)
    max_iter=1200,          # Maximum number of iterations for optimization
    random_state=42        # Random seed for reproducibility
    ).fit(X, y)

with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
