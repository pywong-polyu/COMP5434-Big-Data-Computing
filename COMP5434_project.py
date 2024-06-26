# Import necessary packages
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

# Data Load & Pre-processing
data_root = 'CORD_19/'  
metadata_path = os.path.join(data_root, 'meta_10k.csv')
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str,
    'doi': str
})

print(len(meta_df))
print(meta_df.head())

# Display the info of the dataframe
meta_df.info()

# Function to glob json files
def glob_files(path, f_type=".json"):
    dst = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(f_type):
                dst.append(os.path.join(root, f))
    return dst

# Glob json files
json_dir = "subset/document_parses/pdf_json/"
print(json_dir)
json_files = glob_files(json_dir, ".json")
print("Total json files:", len(json_files))

# Class to read JSON files
class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            for entry in content['abstract']:
                self.abstract.append(entry['text'])
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
            self.title = content['metadata']['title']

    def __repr__(self):
        return f"{self.paper_id}: {self.title } : {self.abstract[:200]}... {self.body_text[:200]}..."

# Read the first row
first_row = FileReader(json_files[0])
print(first_row)

# Function to add breaks to text
from tqdm import tqdm

def get_breaks(content, length):
    data = ""
    words = content.split(' ')
    total_chars = 0

    # Add break every length characters
    for i in range(len(words)):
        total_chars += len(words[i])
        if total_chars > length:
            data = data + "<br>" + words[i]
            total_chars = 0
        else:
            data = data + " " + words[i]
    return data

# Dictionary to hold data
dict_ = {'paper_id': [], 'doi':[], 'abstract': [], 'body_text': [],
         'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}

# Iterate through json files and extract content
for idx, entry in tqdm(enumerate(json_files), total=len(json_files)):
    try:
        content = FileReader(entry)
    except Exception as e:
        continue  # Invalid paper format, skip

    # Get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    # No metadata, skip this paper
    if len(meta_data) == 0:
        continue
    if len(content.body_text) == 0:
        continue
    dict_['abstract'].append(content.abstract)
    dict_['paper_id'].append(content.paper_id)
    dict_['body_text'].append(content.body_text)
    # Also create a column for the summary of abstract to be used in a plot
    if len(content.abstract) == 0:
        # No abstract provided
        dict_['abstract_summary'].append("Not provided.")
    elif len(content.abstract.split(' ')) > 100:
        # Abstract provided is too long for plot, take first 100 words append with ...
        info = content.abstract.split(' ')[:100]
        summary = get_breaks(' '.join(info), 40)
        dict_['abstract_summary'].append(summary + "...")
    else:
        # Abstract is short enough
        summary = get_breaks(content.abstract, 40)
        dict_['abstract_summary'].append(summary)

    # Get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    try:
        # If more than one author
        authors = meta_data['authors'].values[0].split(';')
        if len(authors) > 2:
            # More than 2 authors, may be problem when plotting, so take first 2 append with ...
            dict_['authors'].append(get_breaks('. '.join(authors), 40))
        else:
            # Authors will fit in plot
            dict_['authors'].append(". ".join(authors))
    except Exception as e:
        # If only one author - or Null value
        dict_['authors'].append(meta_data['authors'].values[0])

    # Add the title information, add breaks when needed
    try:
        title = get_breaks(meta_data['title'].values[0], 40)
        dict_['title'].append(title)
    # If title was not provided
    except Exception as e:
        dict_['title'].append(meta_data['title'].values[0])

    # Add the journal information
    dict_['journal'].append(meta_data['journal'].values[0])

    # Add doi
    dict_['doi'].append(meta_data['doi'].values[0])

# Create a DataFrame
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'doi', 'abstract', 'body_text',
                                        'authors', 'title', 'journal', 'abstract_summary'])
print(df_covid.head())

# Display the info of the dataframe
df_covid.info()

# Drop NaN values
df = df_covid
df.dropna(inplace=True)
df.info()

# Install langdetect
subprocess.run(['pip', 'install', 'langdetect'])

# Language detection
from langdetect import detect
from langdetect import DetectorFactory

# Set seed
DetectorFactory.seed = 0

# Hold label - language
languages = []

# Go through each text
for ii in tqdm(range(0, len(df))):
    # Split by space into list, take the first x index, join with space
    text = df.iloc[ii]['body_text'].split(" ")

    lang = "en"
    try:
        if len(text) > 50:
            lang = detect(" ".join(text[:50]))
        elif len(text) > 0:
            lang = detect(" ".join(text[:len(text)]))
    # Beginning of the document was not in a good format
    except Exception as e:
        all_words = set(text)
        try:
            lang = detect(" ".join(all_words))
        # Let's see if we can find any text in abstract...
        except Exception as e:
            try:
                # Let's try to label it through the abstract then
                lang = detect(df.iloc[ii]['abstract_summary'])
            except Exception as e:
                lang = "unknown"
                pass

    # Get the language
    languages.append(lang)

# Print language statistics
from pprint import pprint

languages_dict = {}
for lang in set(languages):
    languages_dict[lang] = languages.count(lang)

print("Total: {}\n".format(len(languages)))
pprint(languages_dict)

# Filter English language papers
df['language'] = languages
df = df[df['language'] == 'en']
df.info()

# Histogram of year / journal
# Convert publish_time to datetime
meta_df['publish_time'] = pd.to_datetime(meta_df['publish_time'], errors='coerce')

# Drop rows with NaT values in publish_time
meta_df = meta_df.dropna(subset=['publish_time'])

# Plot histogram of publication years
plt.figure(figsize=(12, 6))
meta_df['publish_time'].dt.year.value_counts().sort_index().plot(kind='bar')
plt.title('Histogram of Publication Years')
plt.xlabel('Year')
plt.ylabel('Number of Papers')
plt.show()

# Plot histogram of journals
plt.figure(figsize=(12, 6))
df['journal'].value_counts().head(20).plot(kind='bar')
plt.title('Top 20 Journals by Number of Papers')
plt.xlabel('Journal')
plt.ylabel('Number of Papers')
plt.xticks(rotation=90)
plt.show()