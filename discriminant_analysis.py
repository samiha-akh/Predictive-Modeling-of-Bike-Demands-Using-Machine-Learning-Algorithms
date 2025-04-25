import pandas as pd
import numpy as np
import sklearn.discriminant_analysis as skl_da
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('training_data_fall2024.csv', sep=',') 

df['increase_stock'] = df['increase_stock'].map({
    'low_bike_demand' : 0,
    'high_bike_demand' : 1

})

# Separate features and target variable
x = df.drop(columns=['increase_stock'])
y = df['increase_stock']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=1)

# Define the hyperparameter grid for LDA
param_grid = [
    {'solver': ['svd']},  # No shrinkage allowed for 'svd'
    {'solver': ['lsqr', 'eigen'], 'shrinkage': ['auto', 0.1, 0.5, 1.0]},  # Shrinkage for 'lsqr' and 'eigen'
]

# Grid search with cross-validation 
model_lda = skl_da.LinearDiscriminantAnalysis()
grid_search = GridSearchCV(model_lda, param_grid, cv=5)

# Fit grid search
grid_search.fit(x_train, y_train)

print("Best Parameters:", grid_search.best_params_)
best_lda = grid_search.best_estimator_

y_pred = best_lda.predict(x_test)

# Evaluate performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification report \n", classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

#Define the hyperparameter grid for QDA
param_grid = {
    'reg_param': [0.0, 0.1, 0.5, 1.0],
    'tol': [1e-4, 1e-3, 1e-2]
}

print("Best Parameters:", grid_search.best_params_)
best_qda = grid_search.best_estimator_

y_pred = best_qda.predict(x_test)

# Evaluate performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification report \n", classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
