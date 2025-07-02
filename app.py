import streamlit as st
import pickle
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Text preprocessing
def transform_text(text):
    text = text.lower()
    doc = nlp(text)

    y = []
    for token in doc:
        if token.is_alpha and token.text not in STOP_WORDS:
            y.append(token.lemma_)

    return " ".join(y)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
