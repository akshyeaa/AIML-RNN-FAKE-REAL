import streamlit as st
import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import google.generativeai as genai

MODEL_PATH = "fake_news_rnn.h5"
TOKENIZER_PATH = "tokenizer.pickle"
MAX_LEN = 150

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

@st.cache_resource
def load_rnn_resources():
    model = load_model(MODEL_PATH, compile=False)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


rnn_model, tokenizer = load_rnn_resources()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

def rnn_predict(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN)

    prob = rnn_model.predict(pad, verbose=0)[0][0]
    label = "Real News" if prob > 0.5 else "Fake News"
    confidence = prob if prob > 0.5 else (1 - prob)

    return label, confidence

def gemini_predict(text, api_key):
    genai.configure(api_key=api_key)

    gemini_model = genai.GenerativeModel(
        model_name="models/gemini-2.5-flash"
    )

    prompt = f"""
You are a news verification assistant.

Classify the following content as Real News or Fake News.
Provide a short explanation (2–3 lines).

Content:
{text}

Respond in the following format:
Label: <Real/Fake>
Explanation: <short explanation>
"""

    response = gemini_model.generate_content(prompt)
    return response.text.strip()

st.title("📰 Fake News Detection System")
st.caption(
    "A hybrid system using Deep Learning (RNN) and optional Generative AI (Gemini Flash) "
    "for enhanced news verification."
)

st.markdown("---")

with st.expander("🔑 Optional: Add your Gemini API Key"):
    user_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Paste your Gemini API key here"
    )

news_text = st.text_area(
    "Enter news content (headline or article)",
    height=220,
    placeholder="Paste the news article here..."
)

if st.button("🔍 Analyze News", use_container_width=True):
    if news_text.strip() == "":
        st.warning("Please enter some news text to analyze.")
    else:
        with st.spinner("Analyzing content..."):
            rnn_label, rnn_conf = rnn_predict(news_text)

            gemini_output = None
            if user_api_key.strip() != "":
                try:
                    gemini_output = gemini_predict(news_text, user_api_key)
                except Exception as e:
                    gemini_output = f"⚠️ Gemini Error: {str(e)}"

        st.markdown("---")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.subheader("🧠 RNN Analysis")

            if rnn_label == "Real News":
                st.success("**Result: Real News**")
            else:
                st.error("**Result: Fake News**")

            st.metric("Confidence", f"{rnn_conf:.2f}")

            if len(news_text.split()) < 40:
                st.warning(
                    "⚠️ RNN performs best on long-form news articles."
                )

        with col2:
            st.subheader("🤖 Gemini Flash Analysis")

            if gemini_output:
                st.markdown(gemini_output)
            else:
                st.info(
                    "Gemini analysis not available.\n\n"
                    "Add your Gemini API key above to enable AI explanations."
                )

st.markdown("---")
st.caption(
    "ℹ️ The RNN model performs linguistic analysis on news articles. "
    "Gemini provides optional reasoning-based explanations using the user's API key."
)


