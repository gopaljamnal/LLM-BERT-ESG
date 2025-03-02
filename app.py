from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# # Load the
#  pre-trained LLM model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = AutoModelForSeq2SeqLM.from_pretrained("gpt2")
from transformers import AutoModelForSeq2SeqLM, GPT2Config

config = GPT2Config.from_pretrained("gpt2")
model = AutoModelForSeq2SeqLM.from_pretrained("gpt2", config=config)

@app.route('/generate_response', methods=['POST'])
def generate_response():
    prompt = request.json['prompt']

    # Preprocess the prompt if necessary
    # ...

    # Generate a response
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)