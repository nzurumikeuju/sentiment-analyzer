import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample training data
texts = [
    "I love this movie!", "What a fantastic experience.",
    "This was okay, not great.", "It was boring and too long.",
    "Absolutely amazing!", "Worst day ever.",
    "I feel neutral about this.", "This is terrible.", "Quite good!",
    "I didn't like it.", "It made me smile.", "I’m not sure how I feel."
]

labels = [
    "positive", "positive",
    "neutral", "negative",
    "positive", "negative",
    "neutral", "negative", "positive",
    "negative", "positive", "neutral"
]

# Create the model pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(texts, labels)

# Streamlit UI
st.title("🤖 How Does AI Know How You Feel?")
st.write("Enter a sentence below and find out if the AI thinks it's **positive**, **negative**, or **neutral**!")

user_input = st.text_input("Type your sentence here:")

if user_input:
    prediction = model.predict([user_input])[0]
    st.write(f"### 🔍 Prediction: {prediction.capitalize()}")

    if prediction == "positive":
        st.success("Great! The AI thinks this is a positive message 😊")
    elif prediction == "negative":
        st.error("Oh no! The AI picked up some negativity 😞")
    else:
        st.info("Hmm, this seems neutral 😐")

st.markdown("---")
st.caption("Demo by Dr. Obianuju Nzurumike – Sentiment Analysis with Scikit-learn")