import csv
import re

from collections import defaultdict
from functools import reduce
from data_preprocessing import *



def load_documents_mr(data_path):
    # Sample data: list of documents (each document is a string)
    documents_MR = []
    with open(data_path, 'r', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            documents_MR.append(row[4])  # Assuming the 4th column is at index 3 (0-based indexing)
            documents_MR.append(row[9])  # Assuming the 9th column is at index 4 (0-based indexing)

    return documents_MR


# Mapper function
def map_function(document):
    # Split the document into words and assign an ID to each word
    for idx, word in enumerate(document.split(), start=1):
      yield (word.lower(), 1)

def map_reduce(documents_MR):
    
    # Specify the words to include in the stoplist
    custom_stop_words = load_stopword()

    # Step 1: Map phase
    mapped = []
    for document in documents_MR:
        mapped.extend(map_function(document))

    # Step 2: Shuffle and sort phase (group by key)
    shuffled = defaultdict(list)
    for word, count in mapped:
        shuffled[word].append(count)

    # Step 3: Reduce phase
    reduced = {}
    for word, counts in shuffled.items():
        reduced[word] = reduce(lambda x, y: x + y, counts)

    # Step 4: Merge the counts of words with uppercase and lowercase versions together
    merged_counts = defaultdict(int)
    for word, count in reduced.items():
        merged_word = re.sub(r'[^a-zA-Z0-9]', '', word)
        # Remove non-alphanumeric characters
        merged_counts[merged_word.lower()] += count

    # Step 5.1: filter out the stop words
    filtered = {word: count for word, count in reduced.items() if word.lower() not in custom_stop_words}

    # Step 5.2: Sort the filtered words by count in descending order
    sorted_words = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    sorted_words_unfilter = sorted(reduced.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_words, sorted_words_unfilter