# Import libraries and Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import pickle
from sklearn.ensemble import GradientBoostingClassifier

dataset = pd.read_csv('diabetes.csv')

# Data Preprocessing
dataset_X = dataset.iloc[:,[1, 2, 3, 4, 5, 6, 7]].values
dataset_Y = dataset.iloc[:,8].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)

dataset_scaled = pd.DataFrame(dataset_scaled)

X = dataset_scaled
Y = dataset_Y

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = dataset['Outcome'] )

# Data Modelling
gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbc.fit(X_train, Y_train)

gbc.score(X_test, Y_test)

Y_pred = gbc.predict(X_test)

# Save and load the model
pickle.dump(gbc, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Assuming you have your true labels in `Y_test` and predictions in `Y_pred`
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix: \n{conf_matrix}")

# Predict using the model
# Assuming missing values are replaced by the mean of the available data for those features
#mean_values = dataset.mean()
#input_data = [86, 66, 26.6, 31] + mean_values.tolist()[4:7]
#print(model.predict(sc.transform(np.array([input_data]))))
