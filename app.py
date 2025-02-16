from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Set Flask app with correct template folder
app = Flask(__name__, template_folder='templates')

# Load the model and vectorizer
try:
    with open('model.pkl', 'rb') as model_file, open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        model = pickle.load(model_file)
        tfidf_vectorizer = pickle.load(vectorizer_file)
except Exception as e:
    print("Error loading model/vectorizer:", e)
    model, tfidf_vectorizer = None, None

def detect(input_text):
    """Checks plagiarism using the trained model."""
    if model and tfidf_vectorizer:
        vectorized_text = tfidf_vectorizer.transform([input_text])
        result = model.predict(vectorized_text)
        return "Plagiarism Detected" if result[0] == 1 else "No Plagiarism Detected"
    return "Error: Model not loaded."

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    input_text = request.form['text']
    detection_result = detect(input_text)
    return render_template('index.html', result=detection_result)

if __name__ == "__main__":
    app.run(debug=True)
