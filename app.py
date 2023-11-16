import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def transform_text(texts):
    texts = texts.lower()
    texts = nltk.word_tokenize(texts)

    y = []
    for i in texts:
        if i.isalnum():
            y.append(i)

    texts = y[:]
    y.clear()

    for i in texts:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    texts = y[:]
    y.clear()

    for i in texts:
        y.append(ps.stem(i))

    return " ".join(y)

tiv = pickle.load(open('vectorizer.pkl', 'rb'))
algo = pickle.load(open('model.pkl', 'rb'))

st.title('Sms Spam Classifier')

text = st.text_area('Enter the message')
if st.button('Predict'):

    # pre process
    new_text = transform_text(text)
    # vectorize
    vector_input = tiv.transform([new_text])
    # predict
    ans = algo.predict(vector_input)[0]
    # display
    if ans == 1 :
        st.header('SPAM!')
    else :
        st.header('NOT SPAM!')


