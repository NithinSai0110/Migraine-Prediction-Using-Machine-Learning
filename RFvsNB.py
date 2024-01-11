import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

os.chdir("C:\\Users\\nithi\\OneDrive\\Documents")

# Load the dataset
df2 = pd.read_csv("migrainedata.csv")

# Select the features for the prediction
selected_features = ['Age', 'Duration', 'Intensity', 'Nausea', 'Vomit', 'Phonophobia', 'Photophobia', 'Visual', 'Sensory', 'Vertigo']

# Number of iterations for testing
num_iterations = 20

# Create lists to store accuracy scores for Random Forest and Naive Bayes
rf_accuracy_scores = []
nb_accuracy_scores = []

# Initialize Random Forest and Naive Bayes classifiers
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
nb_clf = GaussianNB()

for iteration in range(num_iterations):
    # Set a different random seed for each iteration
    random_seed = 42 + iteration

    # Split the dataset into training and testing sets with a new random seed
    X_train, X_test, y_train, y_test = train_test_split(df2[selected_features], df2["Type"], test_size=0.2, random_state=random_seed)

    # Train Random Forest and Naive Bayes classifiers
    rf_clf.fit(X_train, y_train)
    nb_clf.fit(X_train, y_train)

    # Predict using Random Forest and Naive Bayes
    y_pred_rf = rf_clf.predict(X_test)
    y_pred_nb = nb_clf.predict(X_test)

    # Calculate accuracy scores for Random Forest and Naive Bayes
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    nb_accuracy = accuracy_score(y_test, y_pred_nb)

    # Append accuracy scores to the respective lists
    rf_accuracy_scores.append(rf_accuracy)
    nb_accuracy_scores.append(nb_accuracy)

# Print the accuracy scores for all iterations
print("Random Forest Accuracy Scores:")
for i, score in enumerate(rf_accuracy_scores):
    print(f"Iteration {i+1}: {score:.2f}")

print("\nNaive Bayes Accuracy Scores:")
for i, score in enumerate(nb_accuracy_scores):
    print(f"Iteration {i+1}: {score:.2f}")

# Input new values for prediction
Age = float(input("Enter the value for Age: "))
Duration = float(input("Enter the value for Duration: "))
Intensity = float(input("Enter the value for Intensity: "))
print("Input 1 if Symptom is experienced else input 0")
Nausea = float(input("Enter the value for Nausea: "))
Vomit = float(input("Enter the value for Vomit: "))
Phonophobia = float(input("Enter the value for Phonophobia: "))
Photophobia = float(input("Enter the value for Photophobia: "))
Visual = float(input("Enter the value for Visual: "))
Sensory = float(input("Enter the value for Sensory: "))
Vertigo = float(input("Enter the value for Vertigo: "))

# Create user_data with feature names
user_data = np.array([[Age, Duration, Intensity, Nausea, Vomit, Phonophobia, Photophobia, Visual, Sensory, Vertigo]], dtype=np.float64)
user_data = pd.DataFrame(user_data, columns=selected_features)

# Predict "Type" using Random Forest and Naive Bayes classifiers
rf_prediction = rf_clf.predict(user_data)
nb_prediction = nb_clf.predict(user_data)

# Print the predictions
print("Prediction Using Random Forest: ", rf_prediction[0])
print("Prediction Using Naive Bayes: ", nb_prediction[0])