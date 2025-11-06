import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import  PorterStemmer
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

# Change to lowercase
def transform_text(Message):
  Message = Message.lower()
  Message = nltk.word_tokenize(Message)

# remove symbol
  y = []
  for i in Message:
    if i.isalnum():
      y.append(i)

  Message = y[:]
  y.clear()

# remove stopwords
  for i in Message:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

# change to base word
  Message = y[:]
  y.clear()

  for i in Message:
    y.append(ps.stem(i))

  return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Spam Email Classifier')
input_sms = st.text_area('Enter your message')

if st.button('Predict'):
    transform_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transform_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header('You have spam')
    else:
        st.header('You do not have spam')
