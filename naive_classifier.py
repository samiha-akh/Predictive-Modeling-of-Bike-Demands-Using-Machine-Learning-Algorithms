import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    X = df.drop(columns=['increase_stock'])
    y = df['increase_stock'].map({'low_bike_demand': 0, 'high_bike_demand': 1})

    return X, y

def split_and_scale_data(X, y, test_size=0.20, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

def naive_classifier(y_test):
    np.random.seed(1)
    random_predictions = np.random.choice([0, 1], size=len(y_test))
    return random_predictions

def evaluate_naive_classifier(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

def main():
    X, y = load_and_preprocess_data("training_data_fall2024.csv")
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale_data(X, y)
    random_predictions = naive_classifier(y_test)
    print("Naive Classifier Performance:")
    evaluate_naive_classifier(y_test, random_predictions)

if __name__ == "__main__":
    main()