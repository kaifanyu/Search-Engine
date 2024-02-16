import os
import json
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

DIRECTORY = 'WEBPAGES_RAW'
OUTPUT = 'inverse_index.json'

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def load_bookkeeping(bookkeeping_path):
    with open(bookkeeping_path, 'r') as file:
        return json.load(file)
    
def tokenize(text):
    text = text.lower()  # text to lowercase
    tokens = word_tokenize(text)  # tokenize
    #  lemmatization and remove stopwords
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalpha()]
    return lemmatized_tokens


def parse_and_tokenize(base_path):
    inverted_index = defaultdict(lambda: defaultdict(lambda: {'tfidf': 0}))
    bookkeeping = load_bookkeeping(os.path.join(base_path, 'bookkeeping.json'))

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path) or folder in ['bookkeeping.json', 'bookkeeping.tsv']:
            continue
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            doc_id = f"{folder}/{file}"
            if doc_id not in bookkeeping:  # if not in bookeeping skip it
                continue
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as html_file:
                soup = BeautifulSoup(html_file, 'lxml-xml')
                text = soup.get_text()
                tokens = tokenize(text)
                
                for token in tokens:
                    inverted_index[token][doc_id]['tfidf'] = 0  # TF-IDF to 0 to be implemented later
                    
    return inverted_index

if __name__ == "__main__":
    base_path = DIRECTORY 
    inverted_index = parse_and_tokenize(base_path)
    

    with open('inverted_index.json', 'w') as f:
        json.dump(inverted_index, f, indent=4)