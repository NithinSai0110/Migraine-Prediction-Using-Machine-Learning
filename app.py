from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the saved model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    age = float(request.form['Age'])
    duration = float(request.form['Duration'])
    intensity = float(request.form['Intensity'])
    nausea = float(request.form['Nausea'])
    vomit = float(request.form['Vomit'])
    phonophobia = float(request.form['Phonophobia'])
    photophobia = float(request.form['Photophobia'])
    visual = float(request.form['Visual'])
    sensory = float(request.form['Sensory'])
    vertigo = float(request.form['Vertigo'])

    # Arrange input into a NumPy array
    input_data = np.array([[age, duration, intensity, nausea, vomit,
                            phonophobia, photophobia, visual, sensory, vertigo]])

    # Predict using the loaded model
    prediction = model.predict(input_data)[0]

    # Show result
    return render_template('result.html', prediction=f"Migraine Type: {prediction}")

if __name__ == '__main__':
    app.run(debug=True)
