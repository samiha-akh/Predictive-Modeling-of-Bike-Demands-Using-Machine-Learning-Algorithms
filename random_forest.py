import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

# Load the dataset
data = pd.read_csv('I:/UU/Statistical Machine Learning/Project/training_data_fall2024.csv')

# Map low bike demand to 0, and high map demand to 1
data['increase_stock'] = data['increase_stock'].map({
    'low_bike_demand' : 0,
    'high_bike_demand' : 1

})

# Drop row with missing values
data.dropna(inplace=True)

# Split features and target
X = data.drop(columns=['increase_stock'])
y = data['increase_stock']

# Hyperparameter tuning using Grid search
param_grid = {
     'n_estimators': [100, 200],           
    'max_depth': [None, 10, 20, 30],         
    'min_samples_split': [2, 5, 10],                 
    'min_samples_leaf': [1, 2, 4]      
}


grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)

# Split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

rf_classifier = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = rf_classifier.predict(X_test)

print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
