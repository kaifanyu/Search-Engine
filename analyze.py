import json
import os

def load_bookkeeping(bookkeeping_path):
    with open(bookkeeping_path, 'r') as file:
        return json.load(file)
    
def analyze_inverted_index(index_path):
    # load the inverted index
    with open(index_path, 'r') as f:
        inverted_index = json.load(f)
    
    # calculate the number of unique words
    unique_words = len(inverted_index)
    
    # calculate the number of documents and prepare document list
    document_set = set()
    for word in inverted_index:
        document_set.update(inverted_index[word].keys())
    number_of_documents = len(document_set)
    
    # calculate the size on disk
    index_size_kb = os.path.getsize(index_path) / 1024  # Size in KB
    
    return unique_words, number_of_documents, index_size_kb

def retrieve_urls_for_query(inverted_index, bookkeeping, query):    
    # collect document IDs for all tokens in the query
    query = query.lower()
    document_ids = set()
    if query in inverted_index:
        document_ids.update(inverted_index[query].keys())
    
    urls = []
    for doc_id in document_ids:
        if doc_id in bookkeeping:
            url = bookkeeping[doc_id]
            tfidf = inverted_index[query][doc_id]['tfidf']  # Retrieve the TF-IDF score
            urls.append((tfidf, url))
    
    #sort by tfid all 0 rn so dont sort
    #urls.sort(key=lambda x: x[1], reverse=True)

    return len(urls), urls[:20]  # len + first 20


if __name__ == "__main__":
    index_path = 'inverted_index.json'
    bookkeeping_path = 'WEBPAGES_RAW/bookkeeping.json'
    
    # oad bookkeeping
    bookkeeping = load_bookkeeping(bookkeeping_path)
    
    queries = ["Informatics", "Mondego", "Irvine"]
    unique_words, number_of_documents, index_size_kb = analyze_inverted_index(index_path)

    with open(index_path, 'r') as f:
        inverted_index = json.load(f)

    with open('analytics.txt', 'w') as analytics:
        # Print the index statistics to the file
        analytics.write(f"Unique Words: {unique_words}\n")
        analytics.write(f"Number of Documents: {number_of_documents}\n")
        analytics.write(f"Index Size on Disk: {index_size_kb:.2f} KB")
        
        # Process each query and print the results to the file
        for query in queries:
            amount, urls = retrieve_urls_for_query(inverted_index, bookkeeping, query)
            analytics.write(f"\n\nQuery: {query}\nNumber of URLs retrieved: {amount}")
            for url in urls:
                analytics.write(f"\n{url}")
