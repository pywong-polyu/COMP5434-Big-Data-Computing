
import os
import json
import glob
import nltk
import string
from tqdm import tqdm
import pandas as pd
from langdetect import detect
from langdetect import DetectorFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings('ignore')


def load_meta_data_from_local(data_root):
    metadata_path = os.path.join(data_root, 'meta_10k.csv')
    meta_df = pd.read_csv(metadata_path, index_col=0,converters={
        'pubmed_id': str,
        'Microsoft Academic Paper ID': str,
        'doi': str
    })

    print('Meta data size:',len(meta_df))
    
    return meta_df


def glob_files(path, f_type=".json"):
    dst = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(f_type):
                dst.append(os.path.join(root, f))
    return dst

def glob_json_files_from_local(data_root):
    # glob json files
    json_dir = os.path.join(data_root, "subset","subset","document_parses","pdf_json")
    print(json_dir)
    json_files = glob_files(json_dir, ".json")

    print("total json files:", len(json_files))
    
    return json_files


# Function to truncate text
def truncate_text(content, length):
    words = content.split(' ')
    if len(words) > length:
        return ' '.join(words[:length]) + "..."
    else:
        return ' '.join(words)

def get_full_data_df(meta_df,json_files):

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
            # No abstract provided, use truncated body text
            summary = truncate_text(content.body_text, 100)
            dict_['abstract_summary'].append(summary)
            
        elif len(content.abstract.split(' ')) > 100:
            # Abstract provided is too long for plot, take first 100 words append with ...
            info = content.abstract.split(' ')[:100]
            summary = ' '.join(info) + "..."
            dict_['abstract_summary'].append(summary)
        else:
            # Abstract is short enough
            summary = content.abstract
            dict_['abstract_summary'].append(summary)

        # Get metadata information
        meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

        try:
            # If more than one author
            authors = meta_data['authors'].values[0].split(';')
            if len(authors) > 2:
                # More than 2 authors, may be problem when plotting, so take first 2 append with ...
                dict_['authors'].append('. '.join(authors[:2]) + '...')
            else:
                # Authors will fit in plot
                dict_['authors'].append(". ".join(authors))
        except Exception as e:
            # If only one author - or Null value
            dict_['authors'].append(meta_data['authors'].values[0])

        # Add the title information, add breaks when needed
        try:
            title = truncate_text(meta_data['title'].values[0], 40)
            dict_['title'].append(title)
        # If title was not provided
        except Exception as e:
            dict_['title'].append(meta_data['title'].values[0])

        # Add the journal information
        dict_['journal'].append(meta_data['journal'].values[0])

        # Add doi
        dict_['doi'].append(meta_data['doi'].values[0])

    # Create a DataFrame
    df = pd.DataFrame(dict_, columns=['paper_id', 'doi', 'abstract', 'body_text',
                                            'authors', 'title', 'journal', 'abstract_summary'])

    df = df.dropna()
    
    return df


def detect_language(df):
    
    # set seed
    DetectorFactory.seed = 0

    # hold label - language
    languages = []

    # go through each text
    for ii in tqdm(range(0,len(df))):
        # split by space into list, take the first x intex, join with space
        text = df.iloc[ii]['body_text'].split(" ")

        lang = "en"
        try:
            if len(text) > 50:
                lang = detect(" ".join(text[:50]))
            elif len(text) > 0:
                lang = detect(" ".join(text[:len(text)]))
        # ught... beginning of the document was not in a good format
        except Exception as e:
            all_words = set(text)
            try:
                lang = detect(" ".join(all_words))
            # what!! :( let's see if we can find any text in abstract...
            except Exception as e:

                try:
                    # let's try to label it through the abstract then
                    lang = detect(df.iloc[ii]['abstract_summary'])
                except Exception as e:
                    lang = "unknown"
                    pass

        # get the language
        languages.append(lang)
        
        
    # languages_dict = {}
    # for lang in set(languages):
    #     languages_dict[lang] = languages.count(lang)

    # print("Total: {}\n".format(len(languages)))
    # print(languages_dict)
        
    return languages

def filter_languages_by_eng(df,languages):
    df['language'] = languages
    df = df[df['language'] == 'en']
    return df


def load_data_from_local(data_root):
    
    # Modify these 2 lines if not loading data from local
    meta_df = load_meta_data_from_local(data_root)
    json_files = glob_json_files_from_local(data_root)
    
    df = get_full_data_df(meta_df,json_files)
    
    # Filter English paper only
    languages = detect_language(df)
    df = filter_languages_by_eng(df,languages)
    
    # Text processing
    df = remove_stopwords(df)
    
    df = df.reset_index().rename(columns={'index':'doc_num'})

    return df


def load_stopword():
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = list(stopwords.words('english'))
    punctuations = list(string.punctuation)
    stop_words += punctuations
    return stop_words



def clean_text(text):
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    return ' '.join(words)

# def spacy_tokenizer(sentence):
#     mytokens = nlp(sentence)
#     mytokens = [word.lemma_.lower().strip() if word.lemma != '-PORN-' else word.lower_ for word in mytokens]
#     mytokens = [word for word in mytokens if word not in stopwords and word not in punctuations]
#     mytokens = " ".join([i for i in mytokens])
#     return mytokens


# df['procesed_text'] = df['body_text'].progress_apply(spacy_tokenizer)

def remove_stopwords(df):
    # Remove stopwords
    stop_words = load_stopword()
    
    for word in stop_words:
        df['processed_text'] = df['body_text'].str.lower().str.replace(word,'')
    
    # Apply text cleaning to body_text and generate processed_test
    df['processed_text'] = df['processed_text'].apply(clean_text)
    
    return df


# def tokenize_text(df):
#     # Tokenize body text
#     # df['tokenized_text'] = df['processed_text'].apply(lambda x: vectorizer.fit_transform(x))
#     df['tokenized_text'] = TfidfVectorizer.fit_transform(df['processed_text'].values)

#     return df


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

            #dict_keys(['paper_id', 'metadata', 'abstract', 'body_text',
            #'bib_entries', 'ref_entries', 'back_matter'])


    def __repr__(self):
        return f"{self.paper_id}: {self.title } : {self.abstract[:200]}... {self.body_text[:200]}..."

    