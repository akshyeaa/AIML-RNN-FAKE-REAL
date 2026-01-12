# import streamlit as st
# import re
# import pickle
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import google.generativeai as genai

# MODEL_PATH = "fake_news_rnn.h5"
# TOKENIZER_PATH = "tokenizer.pickle"
# MAX_LEN = 150

# st.set_page_config(
#     page_title="Fake News Detection System",
#     page_icon="📰",
#     layout="centered"
# )

# @st.cache_resource
# def load_rnn_resources():
#     model = load_model(MODEL_PATH, compile=False)
#     with open(TOKENIZER_PATH, "rb") as f:
#         tokenizer = pickle.load(f)
#     return model, tokenizer


# rnn_model, tokenizer = load_rnn_resources()

# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z]', ' ', text)
#     return text

# def rnn_predict(text):
#     text = clean_text(text)
#     seq = tokenizer.texts_to_sequences([text])
#     pad = pad_sequences(seq, maxlen=MAX_LEN)

#     prob = rnn_model.predict(pad, verbose=0)[0][0]
#     label = "Real News" if prob > 0.5 else "Fake News"
#     confidence = prob if prob > 0.5 else (1 - prob)

#     return label, confidence

# def gemini_predict(text, api_key):
#     genai.configure(api_key=api_key)

#     gemini_model = genai.GenerativeModel(
#         model_name="models/gemini-2.5-flash"
#     )

#     prompt = f"""
# You are a news verification assistant.

# Classify the following content as Real News or Fake News.
# Provide a short explanation (2–3 lines).

# Content:
# {text}

# Respond in the following format:
# Label: <Real/Fake>
# Explanation: <short explanation>
# """

#     response = gemini_model.generate_content(prompt)
#     return response.text.strip()

# st.title("📰 Fake News Detection System")
# st.caption(
#     "A hybrid system using Deep Learning (RNN) and optional Generative AI (Gemini Flash) "
#     "for enhanced news verification."
# )

# st.markdown("---")

# with st.expander("🔑 Optional: Add your Gemini API Key"):
#     user_api_key = st.text_input(
#         "Gemini API Key",
#         type="password",
#         placeholder="Paste your Gemini API key here"
#     )

# news_text = st.text_area(
#     "Enter news content (headline or article)",
#     height=220,
#     placeholder="Paste the news article here..."
# )

# if st.button("🔍 Analyze News", use_container_width=True):
#     if news_text.strip() == "":
#         st.warning("Please enter some news text to analyze.")
#     else:
#         with st.spinner("Analyzing content..."):
#             rnn_label, rnn_conf = rnn_predict(news_text)

#             gemini_output = None
#             if user_api_key.strip() != "":
#                 try:
#                     gemini_output = gemini_predict(news_text, user_api_key)
#                 except Exception as e:
#                     gemini_output = f"⚠️ Gemini Error: {str(e)}"

#         st.markdown("---")

#         col1, col2 = st.columns(2, gap="large")

#         with col1:
#             st.subheader("🧠 RNN Analysis")

#             if rnn_label == "Real News":
#                 st.success("**Result: Real News**")
#             else:
#                 st.error("**Result: Fake News**")

#             st.metric("Confidence", f"{rnn_conf:.2f}")

#             if len(news_text.split()) < 40:
#                 st.warning(
#                     "⚠️ RNN performs best on long-form news articles."
#                 )

#         with col2:
#             st.subheader("🤖 Gemini Flash Analysis")

#             if gemini_output:
#                 st.markdown(gemini_output)
#             else:
#                 st.info(
#                     "Gemini analysis not available.\n\n"
#                     "Add your Gemini API key above to enable AI explanations."
#                 )

# st.markdown("---")
# st.caption(
#     "ℹ️ The RNN model performs linguistic analysis on news articles. "
#     "Gemini provides optional reasoning-based explanations using the user's API key."
# )

import streamlit as st
import re
import pickle
import numpy as np
import sys
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, InputLayer
import google.generativeai as genai

MODEL_PATH = "fake_news_rnn.h5"
TOKENIZER_PATH = "tokenizer.pickle"
MAX_LEN = 150

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

# Debug info
st.sidebar.write(f"Python version: {sys.version}")
st.sidebar.write(f"Working directory: {os.getcwd()}")
st.sidebar.write(f"Files in directory: {os.listdir('.')}")

@st.cache_resource
def load_rnn_resources():
    """Load RNN model and tokenizer with comprehensive error handling"""
    try:
        # First check if files exist
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found: {MODEL_PATH}")
            st.sidebar.error(f"Looking for: {os.path.abspath(MODEL_PATH)}")
            return None, None
            
        if not os.path.exists(TOKENIZER_PATH):
            st.error(f"❌ Tokenizer file not found: {TOKENIZER_PATH}")
            st.sidebar.error(f"Looking for: {os.path.abspath(TOKENIZER_PATH)}")
            return None, None
        
        # Try multiple loading strategies
        model = None
        tokenizer = None
        
        # Strategy 1: Standard load with custom objects
        try:
            st.sidebar.info("🔄 Trying Strategy 1: Standard load...")
            custom_objects = {'InputLayer': InputLayer, 'Input': Input}
            model = load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
            st.sidebar.success("✅ Model loaded with Strategy 1")
        except Exception as e1:
            st.sidebar.warning(f"Strategy 1 failed: {str(e1)[:100]}")
            
            # Strategy 2: Try loading with safe_mode=False
            try:
                st.sidebar.info("🔄 Trying Strategy 2: safe_mode=False...")
                model = load_model(MODEL_PATH, compile=False, safe_mode=False)
                st.sidebar.success("✅ Model loaded with Strategy 2")
            except Exception as e2:
                st.sidebar.warning(f"Strategy 2 failed: {str(e1)[:100]}")
                
                # Strategy 3: Try tf.keras directly
                try:
                    st.sidebar.info("🔄 Trying Strategy 3: tf.keras...")
                    import tensorflow as tf
                    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                    st.sidebar.success("✅ Model loaded with Strategy 3")
                except Exception as e3:
                    st.sidebar.error(f"Strategy 3 failed: {str(e3)[:100]}")
                    
                    # Strategy 4: Manual architecture recreation (last resort)
                    st.sidebar.info("🔄 Trying Strategy 4: Manual loading...")
                    try:
                        # Try to load as weights only
                        from tensorflow.keras.models import Sequential
                        from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
                        
                        # This is a generic RNN architecture - you may need to adjust
                        model = Sequential([
                            Input(shape=(MAX_LEN,)),
                            Embedding(input_dim=10000, output_dim=128, input_length=MAX_LEN),
                            LSTM(128, return_sequences=True, dropout=0.2),
                            LSTM(64, dropout=0.2),
                            Dense(64, activation='relu'),
                            Dropout(0.3),
                            Dense(1, activation='sigmoid')
                        ])
                        
                        # Try to load weights
                        model.load_weights(MODEL_PATH)
                        st.sidebar.success("✅ Model loaded with Strategy 4 (weights only)")
                    except Exception as e4:
                        st.sidebar.error(f"All strategies failed: {str(e4)[:100]}")
                        return None, None
        
        # Load tokenizer
        try:
            with open(TOKENIZER_PATH, "rb") as f:
                tokenizer = pickle.load(f)
            st.sidebar.success("✅ Tokenizer loaded")
        except Exception as e:
            st.sidebar.error(f"Tokenizer load failed: {str(e)}")
            return model, None
            
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ Critical error in load_rnn_resources: {str(e)}")
        return None, None

# Load resources with fallback
rnn_model, tokenizer = load_rnn_resources()

# Check if resources loaded successfully
if rnn_model is None or tokenizer is None:
    st.warning("""
    ⚠️ **Model or tokenizer failed to load**
    
    This is usually due to:
    1. Missing model files on Streamlit Cloud
    2. Version incompatibility (TensorFlow/Keras version mismatch)
    3. Corrupted model files
    
    **Quick fixes:**
    1. Make sure `fake_news_rnn.h5` and `tokenizer.pickle` are in your GitHub repo
    2. Update `requirements.txt` with exact versions
    3. Re-save your model locally with: `model.save('fake_news_rnn.keras')`
    
    The app will continue with limited functionality.
    """)
    
    # Create dummy model and tokenizer for demo
    class DummyModel:
        def predict(self, x, verbose=0):
            return np.array([[0.5]])  # Return 50% probability
    
    class DummyTokenizer:
        def texts_to_sequences(self, texts):
            return [[1, 2, 3]]  # Dummy sequence
    
    rnn_model = DummyModel() if rnn_model is None else rnn_model
    tokenizer = DummyTokenizer() if tokenizer is None else tokenizer

def clean_text(text):
    """Clean input text"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def rnn_predict(text):
    """Make prediction using RNN model"""
    try:
        text = clean_text(text)
        
        # Handle empty text after cleaning
        if not text:
            return "Inconclusive", 0.5
        
        seq = tokenizer.texts_to_sequences([text])
        
        # Handle empty sequences
        if not seq or len(seq[0]) == 0:
            return "Inconclusive", 0.5
            
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        
        # Check if we're using dummy model
        if hasattr(rnn_model, '__class__') and 'DummyModel' in str(rnn_model.__class__):
            prob = 0.5
        else:
            prob = rnn_model.predict(pad, verbose=0)[0][0]
        
        # Convert to label and confidence
        label = "Real News" if prob > 0.5 else "Fake News"
        confidence = prob if prob > 0.5 else (1 - prob)
        
        return label, confidence
        
    except Exception as e:
        st.sidebar.error(f"Prediction error: {str(e)}")
        return "Error", 0.5

def gemini_predict(text, api_key):
    """Make prediction using Gemini"""
    try:
        genai.configure(api_key=api_key)
        
        # Updated model name for latest API
        try:
            gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
        except:
            gemini_model = genai.GenerativeModel(model_name="models/gemini-2.0-flash-exp")
        
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
        
    except Exception as e:
        return f"⚠️ Gemini Error: {str(e)}"

# Main UI
st.title("📰 Fake News Detection System")
st.caption(
    "A hybrid system using Deep Learning (RNN) and optional Generative AI (Gemini Flash) "
    "for enhanced news verification."
)

st.markdown("---")

# API Key section
with st.expander("🔑 Optional: Add your Gemini API Key", expanded=False):
    user_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Paste your Gemini API key here (optional)",
        help="Get your API key from https://makersuite.google.com/app/apikey"
    )

# Text input
news_text = st.text_area(
    "Enter news content (headline or article)",
    height=220,
    placeholder="Paste the news article here...",
    help="Enter at least 100 words for best results"
)

# Analyze button
if st.button("🔍 Analyze News", use_container_width=True, type="primary"):
    if not news_text or news_text.strip() == "":
        st.warning("📝 Please enter some news text to analyze.")
    else:
        with st.spinner("🔄 Analyzing content..."):
            # Show warning if using dummy model
            if hasattr(rnn_model, '__class__') and 'DummyModel' in str(rnn_model.__class__):
                st.warning("⚠️ Using demo mode - RNN model not properly loaded")
            
            # Get predictions
            rnn_label, rnn_conf = rnn_predict(news_text)
            
            gemini_output = None
            if user_api_key and user_api_key.strip() != "":
                try:
                    gemini_output = gemini_predict(news_text, user_api_key)
                except Exception as e:
                    gemini_output = f"⚠️ Gemini Error: {str(e)}"
        
        st.markdown("---")
        
        # Display results in columns
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.subheader("🧠 RNN Analysis")
            
            if rnn_label == "Real News":
                st.success(f"**Result: {rnn_label}**")
                st.balloons()
            elif rnn_label == "Fake News":
                st.error(f"**Result: {rnn_label}**")
            else:
                st.warning(f"**Result: {rnn_label}**")
            
            # Confidence gauge
            st.metric("Confidence", f"{rnn_conf:.2%}")
            
            # Word count warning
            word_count = len(news_text.split())
            if word_count < 50:
                st.warning(f"⚠️ Short text ({word_count} words). For better accuracy, provide longer articles (100+ words).")
        
        with col2:
            st.subheader("🤖 Gemini Flash Analysis")
            
            if gemini_output:
                # Parse Gemini output
                if "Label:" in gemini_output:
                    lines = gemini_output.split('\n')
                    for line in lines:
                        if line.startswith("Label:"):
                            label = line.replace("Label:", "").strip()
                            if "real" in label.lower():
                                st.success(f"**Gemini: {label}**")
                            elif "fake" in label.lower():
                                st.error(f"**Gemini: {label}**")
                            else:
                                st.info(f"**Gemini: {label}**")
                        elif line.startswith("Explanation:"):
                            explanation = line.replace("Explanation:", "").strip()
                            st.info(f"**Explanation:** {explanation}")
                else:
                    st.markdown(gemini_output)
            else:
                st.info(
                    "🤖 **Gemini analysis not enabled**\n\n"
                    "To get AI-powered explanations:\n"
                    "1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)\n"
                    "2. Paste it in the 'Optional: Add your Gemini API Key' section above"
                )

st.markdown("---")

# Footer with troubleshooting info
with st.expander("🛠️ Troubleshooting & Info", expanded=False):
    st.markdown("""
    ### **Common Issues & Solutions:**
    
    **❌ Model not loading on Streamlit Cloud:**
    1. Check that `fake_news_rnn.h5` and `tokenizer.pickle` are in your GitHub repository
    2. Update `requirements.txt` with exact versions:
    ```
    streamlit==1.32.0
    tensorflow==2.15.0
    numpy==1.24.3
    pandas==2.1.4
    scikit-learn==1.3.2
    google-generativeai==0.8.0
    ```
    3. Re-save your model locally as `.keras` format for better compatibility:
    ```python
    model.save('fake_news_rnn.keras')  # Instead of .h5
    ```
    
    **❌ Version mismatch errors:**
    - Train and deploy with the same TensorFlow version
    - Use `.keras` format which is more version-stable than `.h5`
    
    **✅ Best practices:**
    - Provide full news articles (not just headlines) for better accuracy
    - Keep your API key secure and don't commit it to GitHub
    - Test locally before deploying to Streamlit Cloud
    """)
    
    # Debug info
    st.markdown("### **Debug Information:**")
    st.code(f"""
    TensorFlow version: {sys.modules['tensorflow'].__version__ if 'tensorflow' in sys.modules else 'Not loaded'}
    Model path: {os.path.abspath(MODEL_PATH)}
    Model exists: {os.path.exists(MODEL_PATH)}
    Tokenizer exists: {os.path.exists(TOKENIZER_PATH)}
    """)

st.caption(
    "ℹ️ The RNN model performs linguistic analysis on news articles. "
    "Gemini provides optional reasoning-based explanations using the user's API key."
)

