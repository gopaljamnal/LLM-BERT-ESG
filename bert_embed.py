from transformers import BertModel, BertTokenizer
import torch
import fitz  # PyMuPDF
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Path to the folder containing PDF files
pdf_folder_path = "data"

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# # Input text
# text = "BERT based architecture has been used to create 768 dimensional vector embeddings."

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text



# # Tokenize the input text and convert it to input IDs and attention mask
# inputs = tokenizer(text, return_tensors="pt")
# input_ids = inputs["input_ids"]
# attention_mask = inputs["attention_mask"]


# Function to generate embedding for a sentence or phrase
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the [CLS] token's embedding as a sentence-level representation
    cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()
    return cls_embedding

# # Get embeddings from BERT model
# with torch.no_grad():
#     outputs = model(input_ids, attention_mask=attention_mask)

# Extract texts and generate embeddings for each sentence in all PDFs
embeddings = []
sentences = []
for filename in os.listdir(pdf_folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_folder_path, filename)
        text = extract_text_from_pdf(file_path)
        # Split the text into sentences (you may want to use nltk or other libraries for more refined sentence splitting)
        for sentence in text.split('. '):  # Simple sentence splitting by periods
            if sentence.strip():  # Skip empty sentences
                embedding = get_embedding(sentence)
                embeddings.append(embedding)
                sentences.append(sentence)

# # The last hidden state (sequence of embeddings) is in `outputs.last_hidden_state`
# # Here, we use the `[CLS]` token's embedding as the sentence-level embedding
# cls_embedding = outputs.last_hidden_state[0, 0, :]  # shape: (768,)
#
# # Convert to numpy array for ease of use
# cls_embedding_np = cls_embedding.numpy()
#
# print("Embedding vector shape:", cls_embedding_np.shape)
# print("Embedding vector:", cls_embedding_np)


# Convert list of embeddings to numpy array for similarity calculation
embeddings = np.array(embeddings)


# Calculate similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Find similar sentences/phrases
similar_threshold = 0.9  # Define similarity threshold
similar_pairs = []
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        if similarity_matrix[i, j] > similar_threshold:
            similar_pairs.append((sentences[i], sentences[j], similarity_matrix[i, j]))

# Display similar sentences/phrases
for s1, s2, score in similar_pairs:
    print(f"Sentence 1: {s1}")
    print(f"Sentence 2: {s2}")
    print(f"Similarity Score: {score:.2f}\n")