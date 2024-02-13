import os
import json
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Constants
DIRECTORY = 'webpages/test'
OUTPUT = 'inverse_index.json'

# # Download stopwords if you haven't done it yet
# nltk.download('stopwords')

# Get English stopwords from NLTK
stop_words = set(stopwords.words('english'))

#lemmatize input
def lemmatize(tokens):
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def get_text(soup):
    # Extract text from title, bold, and h1 tags
    title_text = soup.title.get_text() if soup.title else ''
    bold_text = ' '.join([bold.get_text() for bold in soup.find_all('b')])
    h1_text = ' '.join([h1.get_text() for h1 in soup.find_all('h1')])
    h2_text = ' '.join([h2.get_text() for h2 in soup.find_all('h2')])
    h3_text = ' '.join([h3.get_text() for h3 in soup.find_all('h3')])

    all_text = [title_text, bold_text, h1_text, h2_text, h3_text]
    return all_text


def process_file(file, inverse_index):
    with open(file, 'r', encoding='utf-8') as f:
        # Pass the file handle directly to BeautifulSoup
        soup = BeautifulSoup(f, 'html.parser')

    #used to calculate tfidf
    all_text = get_text(soup)

    # Find all text content within the HTML body
    body_content = soup.body.get_text() if soup.body else ''

    # Use regular expression to extract only words
    words = re.findall(r'\b[a-zA-Z]+\b', body_content)

    # Lowercase everything
    lowercase = [word.lower() for word in words]

    # Tokenize the sentence
    words_tokenized = word_tokenize(' '.join(lowercase))

    # Filter out stopwords
    filtered_words = [word for word in words_tokenized if word.lower() not in stop_words]

    # Lemmatize the tokens
    lemmatized_words = lemmatize(filtered_words)

    # Check if the document is not empty before processing
    if lemmatized_words:
        # Replace backslashes with forward slashes and remove the leading part
        index = file.replace('\\', '/').split('/', 2)[-1]

        # put our words into documents
        documents = [' '.join(lemmatized_words)]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()

        # Store TF-IDF values and additional metadata in a dictionary
        tfidf_dict = {}
        for i, feature in enumerate(feature_names):
            tfidf_dict[feature] = {
                'tfidf': tfidf_matrix[0, i],
                'importance': 'high' if any(feature in sublist for sublist in all_text)  else 'low'
            }

        # Add the information to the inverse index
        inverse_index[index] = tfidf_dict


def create_inverse_index():
    inverse_index = {}
    for root, dirs, files in os.walk(DIRECTORY):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            process_file(file_path, inverse_index)

    return inverse_index


def save_inverse_index_to_json(inverse_index):
    with open(OUTPUT, 'w', encoding='utf-8') as json_file:
        json.dump(inverse_index, json_file, indent=4)

def start():
    inverse_index = create_inverse_index()
    save_inverse_index_to_json(inverse_index)

start()
