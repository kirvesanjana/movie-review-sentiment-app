import streamlit as st
import joblib

# Load the trained model and TF-IDF vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Checker",
    page_icon="🎬"
)

# App title and description
st.title("🎬 Movie Review Sentiment Checker")
st.write("Enter a movie review to predict whether its sentiment is positive or negative.")

# Text input
review = st.text_area("Enter your movie review:")

# Prediction button
if st.button("Analyze Review"):
    if review.strip():
        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)[0]

        if prediction == "positive":
            st.success("Predicted Sentiment: POSITIVE 😊")
        else:
            st.error("Predicted Sentiment: NEGATIVE 😞")
    else:
        st.warning("Please enter a review first.")
