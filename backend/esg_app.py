from flask import Flask, jsonify, request
from flask_cors import CORS
from tika import parser
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Initialize the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nbroad/ESG-BERT")
model = AutoModelForSequenceClassification.from_pretrained("nbroad/ESG-BERT")
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

# PDF Parser Class
class PDFParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw = parser.from_file(self.file_path)
        self.text = self.raw['content']

    def get_text(self):
        return self.text

    def get_text_clean(self):
        text = self.text
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def get_text_clean_list(self):
        text = self.get_text_clean()
        text_list = text.split('.')
        return text_list

# Function to run classification on CSR report
def run_classifier(url):
    pp = PDFParser(url)
    sentences = pp.get_text_clean_list()
    result = classifier(sentences)
    df = pd.DataFrame(result)
    return df.groupby(['label']).mean().sort_values('score', ascending=False).to_dict()

# API endpoint to classify CSR reports
@app.route('/classify', methods=['POST'])
def classify_report():
    try:
        data = request.json
        url = data['url']
        result = run_classifier(url)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Welcome to the ESG Report Classifier!"


if __name__ == "__main__":
    app.run(debug=True)
