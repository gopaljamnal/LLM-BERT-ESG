from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from tika import parser
import re
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("nbroad/ESG-BERT")

model = AutoModelForSequenceClassification.from_pretrained("nbroad/ESG-BERT")

# Create the pipeline for text classification
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)
# Create a Class to parse PDF
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

# Get report from responsibilityreports.com
mcdonalds_url = "https://www.responsibilityreports.com/Click/2534"
pp = PDFParser(mcdonalds_url)
sentences = pp.get_text_clean_list()

print(f"The McDonalds CSR report has {len(sentences):,d} sentences")

result = classifier(sentences)
df = pd.DataFrame(result)

print(df.groupby(['label']).mean().sort_values('score', ascending=False))

# We can also convert the workflow above into a function and can easily compare the scores with other companies'
def run_classifier(url):
    pp = PDFParser(url)
    sentences = pp.get_text_clean_list()
    print(f"The CSR report has {len(sentences):,d} sentences")
    result = classifier(sentences)
    df = pd.DataFrame(result)
    return(df)

# Let's try to look at Amazon
amzn = run_classifier("https://www.responsibilityreports.com/Click/2015")
print(amzn.groupby(['label']).mean().sort_values('score', ascending = False))

# Let's look at another company from a different sector - Newmont Mining
nm = run_classifier("https://www.responsibilityreports.com/Click/1772")
print(nm.groupby(['label']).mean().sort_values('score', ascending = False))
