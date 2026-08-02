# Text cleaning functions for NLP analysis
# USED BY: notebooks/01_sentiment_analysis.ipynb
#          app/app.py (Streamlit application)

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

STOPWORDS = set(stopwords.words('english'))

print("✅ preprocess.py loaded successfully!")


def clean_for_vader(text):

    if not isinstance(text, str):
        return ""

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'<.*?>', ' ', text)

    text = ' '.join(text.split())

    return text.strip()


def clean_for_wordcloud(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'<.*?>', ' ', text)

    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub(r'\d+', '', text)

    text = ' '.join(text.split())

    tokens = word_tokenize(text)

    tokens = [
        word
        for word in tokens
        if word not in STOPWORDS
        and len(word) > 2
    ]

    return ' '.join(tokens)