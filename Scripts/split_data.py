# This script loads the original dataset and splits it into a training set and test set. 

import numpy as np
import pandas as pd

# Read the CSV file.
df = pd.read_csv("voice.csv", header=0)

# Extract the labels into a numpy array. The original labels are text but we convert
# this to numbers: 1 = male, 0 = female.
labels = (df["label"] == "male").values * 1

# labels is a row vector but TensorFlow expects a column vector, so reshape it.
labels = labels.reshape(-1, 1)

# Remove the column with the labels.
del df["label"]

# OPTIONAL: Do additional preprocessing, such as scaling the features.
# for column in df.columns:
#     mean = df[column].mean()
#     std = df[column].std()
#     df[column] = (df[column] - mean) / std

# Convert the training data to a numpy array.
data = df.values
print("Full dataset size:", data.shape)

# Split into a random training set and a test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=123456)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Save the matrices using numpy's native format.
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
