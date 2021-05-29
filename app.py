import numpy as np
from flask import Flask, request, jsonify, render_template
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import joblib


app = Flask(__name__)
model = joblib.load(open('NB_spam_model.pkl', 'rb'))
cv = joblib.load(open('cv.pkl', 'rb'))

ps = PorterStemmer()
sw = set(stopwords.words('english'))


def stemming(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer('\w+')
    tokens = tokenizer.tokenize(sentence) 
    removed_stopwords = [w for w in tokens if w not in sw]
    stemmed_words = [ps.stem(token) for token in removed_stopwords]
    clean_sentence = ' '.join(stemmed_words)
    return clean_sentence

def prepare(messages):
    d = [stemming(messages)]
    return cv.transform(d)

@app.route('/', methods=['POST','GET'])
def home():
    if request.method == 'POST':
        text = request.form['Message']
        final_message = text
        messages = prepare(final_message)
        prediction = model.predict(messages)[0]
        return render_template('index.html', prediction_text='The message is "{}"'.format(prediction))

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)