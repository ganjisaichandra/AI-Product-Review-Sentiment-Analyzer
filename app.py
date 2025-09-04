import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# Load AI Model
sentiment_pipeline = pipeline("sentiment-analysis")

st.set_page_config(page_title="AI Review Analyzer", layout="wide")
st.title("📊 AI Product Review Sentiment Analyzer")
st.write("Upload Amazon/Flipkart reviews and analyze sentiment instantly!")

# Choose input
option = st.radio("Choose Input Type:", ["Single Review", "Upload CSV"])

if option == "Single Review":
    review = st.text_area("Enter a product review:")
    if st.button("Analyze Review"):
        if review.strip():
            result = sentiment_pipeline(review)[0]
            st.success(f"**Sentiment:** {result['label']} (Confidence: {result['score']:.2f})")
        else:
            st.warning("Please enter a review.")

elif option == "Upload CSV":
    file = st.file_uploader("Upload a CSV file with a 'review' column", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write("Preview of Uploaded Data:", df.head())

        if st.button("Analyze All Reviews"):
            sentiments = sentiment_pipeline(list(df["review"]))
            df["Sentiment"] = [s["label"] for s in sentiments]

            # Show results
            st.dataframe(df.head(10))

            # Sentiment distribution
            sentiment_counts = df["Sentiment"].value_counts()
            fig, ax = plt.subplots()
            sentiment_counts.plot(kind="bar", ax=ax, color=["green", "red", "blue"])
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)
