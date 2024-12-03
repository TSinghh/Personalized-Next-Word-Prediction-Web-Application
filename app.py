from flask import Flask, request, render_template, jsonify
import pickle
from collections import defaultdict

# Load the cleaned dialogs data
with open(r"C:\Users\Tanya\OneDrive\Desktop\NLP Project\NLP Project\dialogs_dataset", "rb") as f:
    dialogs = pickle.load(f)

def create_ngram_model(dialogs):
    model = defaultdict(lambda: defaultdict(lambda: 0))
    for sentence in dialogs:
        tokens = sentence.split()
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            model[(w1, w2)][w3] += 1
    # Normalize to probabilities
    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_count
    return model

# Build the trigram model
model = create_ngram_model(dialogs)

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input_text']
    words = input_text.split()
    
    # Ensure at least two words are provided
    if len(words) < 2:
        return jsonify({"error": "Please provide at least two words."})

    # Extract the last two words
    w1, w2 = words[-2], words[-1]

    # Predict the next word
    next_word_probabilities = dict(model[(w1, w2)])
    if not next_word_probabilities:
        return jsonify({"error": "No prediction available for the given input."})

    # Find the most probable next word
    next_word = max(next_word_probabilities, key=next_word_probabilities.get)
    response = {
        "next_word": next_word,
        "probabilities": next_word_probabilities
    }
    return jsonify(response)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
