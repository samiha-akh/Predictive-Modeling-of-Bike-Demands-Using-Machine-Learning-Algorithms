# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Function to load and preprocess the data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Feature Engineering
    df = df.drop(columns=['snow', 'snowdepth'])
    df['low_demand_hour'] = df['hour_of_day'].apply(lambda x: 1 if x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 21, 22, 23] else 0)
    df = df.drop(columns=['hour_of_day', 'day_of_week'])
    df['low_demand_month'] = df['month'].apply(lambda x: 1 if x in [1, 2, 11, 12] else 0)
    df = df.drop(columns=['month'])
    df['good_weather'] = df.apply(lambda row: 1 if row['temp'] > 17 and row['humidity'] < 50 else 0, axis=1)
    df = df.drop(columns=['temp', 'humidity', 'precip', 'visibility', 'dew'])

    # Splitting the data into features (X) and target (y)
    X = df.drop(columns=['increase_stock'])
    y = df['increase_stock']

    return X, y

# Function to split and scale the data
def split_and_scale_data(X, y, test_size=0.20, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Function to perform grid search and find the best model
def train_best_model(X_train, y_train):
    param_grid = {
        'n_neighbors': range(1, 21),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'chebyshev'],
    }

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        verbose=1,  # To show progress
        n_jobs=-1  # Use all available cores
    )

    grid_search.fit(X_train, y_train)
    return grid_search

# Function to evaluate the model on test data
def evaluate_model(model, X_test, y_test):
    test_predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, test_predictions)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_predictions))


def main():
    # Loading and preprocess the data
    X, y = load_and_preprocess_data("training_data_fall2024.csv")
    
    # Splitting and scale the data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # Training the best model using GridSearchCV
    grid_search = train_best_model(X_train_scaled, y_train)
    
    # Getting the best parameters
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"\nBest Parameters: {best_params}")
    print(f"Best Cross-Validation Accuracy: {best_score:.4f}")
    
    # Evaluating the model on the test set
    best_model = grid_search.best_estimator_
    print("\nTest Set Performance:")
    evaluate_model(best_model, X_test_scaled, y_test)


if __name__ == "__main__":
    main()
