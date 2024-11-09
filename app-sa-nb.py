
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


import streamlit as st
import pickle

# Load the saved model
with open('sentiment_pipeline_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Create a title and a text input for the user
st.title("Sentiment Analyzer")
text_input = st.text_input("Enter your text here:")

# When the user presses the button, make a prediction
if st.button("Analyze"):
    if text_input:
        prediction = loaded_model.predict([text_input])[0]
        if prediction == 1:
            st.success("Positive Sentiment")
        else:
            st.error("Negative Sentiment")
    else:
        st.warning("Please enter some text.")
