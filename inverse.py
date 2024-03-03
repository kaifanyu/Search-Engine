import os
import json
import pprint
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from math import log

DIRECTORY = 'webpages/Test'
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
    term_frequency_map = defaultdict(lambda: defaultdict(lambda: {'frequency':0, 'weight': 0}))  # Updated to defaultdict(int)

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path) or folder in ['bookkeeping.json', 'bookkeeping.tsv']:
            continue
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            doc_id = f"{folder}/{file}"
            if doc_id not in bookkeeping:  # if not in bookkeeping, skip it
                continue
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as html_file:
                soup = BeautifulSoup(html_file, 'lxml-xml')
                text = soup.get_text()
                tokens = tokenize(text)

                for token in tokens:
                    term_frequency_map[doc_id][token]['frequency'] += 1  # Increment the frequency
                    inverted_index[token][doc_id]['tfidf'] = 0  # TF-IDF to 0 to be implemented later

            for term, word in term_frequency_map[doc_id].items():
                frequency = word['frequency']
                term_frequency_map[doc_id][term]['weight'] = 1 + log(frequency)

    return inverted_index, term_frequency_map

if __name__ == "__main__":
    base_path = DIRECTORY 
    inverted_index, term_frequency = parse_and_tokenize(base_path)

    for word, value in inverted_index.items():
        for doc_id, subvalue in value.items():
            weight = term_frequency[doc_id][word]['weight']
            print("WEIGHT", weight)
            print(inverted_index[word][doc_id]['tfidf'])
            inverted_index[word][doc_id]['tfidf'] = weight

    with open(OUTPUT, 'w') as f:
        json.dump(inverted_index, f, indent=4)
    with open("term.json", 'w') as f:
        json.dump(term_frequency, f, indent=4)
