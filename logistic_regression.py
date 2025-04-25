import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

# Load the dataset
file_path = './training_data_fall2024 (1).csv'
data = pd.read_csv(file_path)

# Separate features and target variable
X = data.drop(columns=['increase_stock'])
y = data['increase_stock'].map({'low_bike_demand': 0, 'high_bike_demand': 1})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initial Logistic Regression training
log_reg = LogisticRegression( max_iter=5000, random_state=1)
log_reg.fit(X_train_scaled, y_train)

# Evaluate the initial model
y_pred = log_reg.predict(X_test_scaled)

initial_report = classification_report(y_test, y_pred, target_names=['low_bike_demand', 'high_bike_demand'])
print("Initial Model Report:\n", initial_report)
print("Confusion Matrix", confusion_matrix(y_test, y_pred))


# Define hyperparameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  
    'penalty': [ 'l1','l2'],      
    'solver': ['liblinear', 'saga'], 
    'class_weight': [None, 'balanced']
}

# Grid search with cross-validation
grid_search = GridSearchCV(LogisticRegression( max_iter=5000, random_state=1),
                           param_grid,
                           cv=5,
                           scoring='f1_macro',
                           n_jobs=-1)

# Fit grid search
grid_search.fit(X_train_scaled, y_train)

# Best parameters and retraining the model
best_params = grid_search.best_params_
print("Best Parameters:\n", best_params)

best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred_best = best_model.predict(X_test_scaled)
best_report = classification_report(y_test, y_pred_best, target_names=['low_bike_demand', 'high_bike_demand'])
print("Best Model Report:\n", best_report)
print("Confusion Matrix", confusion_matrix(y_test, y_pred_best))