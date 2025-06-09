import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved SVM model
with open('model_svm.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the saved TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Streamlit app UI
st.set_page_config(page_title="Email Spam Detector", layout="centered")
st.title("📬 Email Spam Detection System")
st.write("Enter an email or message below to classify it as **Spam** or **Not Spam**. The system will also display the model's confidence score.")

# Input box for user to enter email content
user_input = st.text_area("Enter the email/message content:", height=200, placeholder="e.g., 'Free entry to win a prize!' or 'Hey, let's meet for lunch.'")

# Predict button
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Vectorize the input text using the loaded vectorizer
        input_tfidf = vectorizer.transform([user_input])

        # Make prediction
        prediction = model.predict(input_tfidf)[0]
        # Get confidence score (maximum probability)
        confidence = model.predict_proba(input_tfidf).max() * 100

        # Display prediction result
        if prediction == 1:
            st.error("🚨 This email is **SPAM**")
        else:
            st.success("✅ This email is **NOT SPAM**")

        # Display confidence score
        st.info(f"📊 **Model Confidence:** {confidence:.2f}%")