import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score

os.chdir("C:\\Users\\nithi\\OneDrive\\Documents")

df2 = pd.read_csv("migrainedata.csv")

selected_features = ['Age', 'Duration', 'Intensity', 'Nausea', 'Vomit', 'Phonophobia', 'Photophobia', 'Visual', 'Sensory', 'Vertigo']

num_iterations = 20

rf_accuracy_scores = []
svm_accuracy_scores = []  

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC()  

for iteration in range(num_iterations):
   
    random_seed = 42 + iteration

    X_train, X_test, y_train, y_test = train_test_split(df2[selected_features], df2["Type"], test_size=0.2, random_state=random_seed)

    rf_clf.fit(X_train, y_train)
    svm_clf.fit(X_train, y_train) 

    y_pred_rf = rf_clf.predict(X_test)
    y_pred_svm = svm_clf.predict(X_test) 

    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    svm_accuracy = accuracy_score(y_test, y_pred_svm) 
   
    rf_accuracy_scores.append(rf_accuracy)
    svm_accuracy_scores.append(svm_accuracy) 

print("Random Forest Accuracy Scores:")
for i, score in enumerate(rf_accuracy_scores):
    print(f"Iteration {i+1}: {score:.2f}")

print("\nSVM Accuracy Scores:")
for i, score in enumerate(svm_accuracy_scores):
    print(f"Iteration {i+1}: {score:.2f}")

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

user_data = np.array([[Age, Duration, Intensity, Nausea, Vomit, Phonophobia, Photophobia, Visual, Sensory, Vertigo]], dtype=np.float64)
user_data = pd.DataFrame(user_data, columns=selected_features)

rf_prediction = rf_clf.predict(user_data)
svm_prediction = svm_clf.predict(user_data)

print("Prediction Using Random Forest: ", rf_prediction[0])
print("Prediction Using SVM: ", svm_prediction[0])
