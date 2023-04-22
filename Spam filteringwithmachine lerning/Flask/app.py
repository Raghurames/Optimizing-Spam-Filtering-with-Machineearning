from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('models/saved/model.joblib')
encoder = joblib.load('models/saved/encoder.joblib')

@app.route('/')
def main():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        message = request.form['submission']
        prediction = model.predict([message])
        classification = encoder.inverse_transform(prediction)

        return render_template('index.html', message=message, classification=classification)

if __name__ == "__main__":
    app.run(debug=True)