import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from streamlit_lottie import st_lottie
import json
import time

# --- Download nltk ---
nltk.download('punkt_tab')
nltk.download('stopwords')
ps = PorterStemmer()

# --- Lottie Animation Loader ---
def load_lottie(path: str):
    with open(path, "r") as f:
        return json.load(f)

# --- Transform text function ---
def transform_text(Message):
    Message = Message.lower()
    Message = nltk.word_tokenize(Message)

    y = [i for i in Message if i.isalnum()]
    Message = y[:]
    y.clear()

    for i in Message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    Message = y[:]
    y.clear()

    for i in Message:
        y.append(ps.stem(i))

    return " ".join(y)

# --- Load model & vectorizer ---
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# --- Page Config ---
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📩",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #e3f2ff 0%, #ffffff 100%);
    }
    .title {
        text-align:center;
        font-size:48px;
        color:#2a65f3;
        font-weight:900;
    }
    .subtitle {
        text-align:center;
        font-size:18px;
        color:#4d4d4d;
        margin-top:-15px;
    }
    .result-box {
        padding:25px;
        border-radius:20px;
        margin-top:20px;
        font-size:22px;
        text-align:center;
        animation: fadeIn 1s ease-in-out;
    }
    @keyframes fadeIn {
        from {opacity:0; transform: translateY(10px);}
        to {opacity:1; transform: translateY(0);}
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 class='title'>📩 Spam Email Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Cek apakah email mengandung spam secara otomatis dengan Machine Learning</p>", unsafe_allow_html=True)
st.write("")

# --- Two columns layout ---
col1, col2 = st.columns([2,1])

with col1:
    input_sms = st.text_area("✉️ Enter Email / Message", height=200, placeholder="Type your email here...")

    if st.button("🚀 Predict", use_container_width=True):
        with st.spinner("Processing..."):
            time.sleep(1)
            transform_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transform_sms])
            result = model.predict(vector_input)[0]

        if result == 1:
            st.markdown("<div class='result-box' style='background:#ffe0e0; color:#b00000;'>🚨 SPAM DETECTED!</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-box' style='background:#e0ffe7; color:#008a2e;'>✅ SAFE - Not Spam</div>", unsafe_allow_html=True)

with col2:
    try:
        animation = load_lottie("mail.json")  # tambahkan file animasi lottie sendiri
        st_lottie(animation, height=250)
    except:
        st.info("You can add animation by placing mail.json next to app.py")

st.markdown("<p style='text-align:center; margin-top:50px;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
