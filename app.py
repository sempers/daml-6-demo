from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__, static_url_path="", static_folder="static")
import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences

import nltk
#nltk.download("punkt")
#nltk.download("wordnet")
stopwords = nltk.corpus.stopwords.words('english')
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
print("nltk loaded")

import pickle

with open('vocab.pckl', 'rb') as f:
	vocab = pickle.load(f)
print("Vocab loaded")

model = tf.keras.models.load_model("best_model_nn.hdf5")
print("neural network loaded")

def tokenize(text):
	return [lemmatizer.lemmatize(w) for w in text.lower().split() if w not in stopwords and w.isalnum()]

def to_sequence(text):
	return np.array([[1] + [vocab[w] if w in vocab else 3 for w in text] + [2]])

@app.route("/", methods=['GET', 'POST'])
def index():
	prediction_message = ""
	text = ""
	if request.method == "POST":
		text = request.form["text"]
		print(f">>> Testing text {text} \n")
		tokens = tokenize(text)
		print(f">>> Tokenization is {tokens}\n")
		seq = to_sequence(tokens)
		print(f">>> Sequence is {seq} \n")
		Y = model.predict(seq)[0][0]
		print(f">>> Prediction is {Y} \n")
		if (Y < 0.5):
			prediction_message = f"negative ({Y:4.2f})"
		else:
			prediction_message = f"positive ({Y:4.2f})"
	return render_template('hello.html', text=text, prediction_message=prediction_message)
	
if __name__ == "__main__":
	app.run(host='0.0.0.0', port=80, debug=False)