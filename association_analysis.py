import pandas as pd
import numpy as np
import scipy.stats as ss
import gensim.downloader as api
import networkx as nx
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori, association_rules 
from mlxtend.preprocessing import TransactionEncoder

from data_preprocessing import *



def get_target_document_index(X,token_list,target_word_list,model_name='glove-wiki-gigaword-100',top_feature=20,top_document=100):
    
    '''
    Get most relateed document with given target word.
    Compare each target word to all token used as feature and find the top related document index.
    Using gensim model for semantic analysis by default using glove-wiki-gigaword-100.
    Return a dictionary of most related documents index for each target word.
    
    Argument:
    
    - X:
    A numpy array of characteristic matrix with row as number of feature and column as number of document. 
    Recommend to use the token count as feature.
    
    - token_list:
    A list of string that use to build the feature of the characteristic matrix.
    
    - target_word_list:
    A list of string that all token compare with.
    Each target word can be any string that is not in the token_list but must be in the model.
    
    - model_name:
    A string of model name that use to give a similarity of score between the target worad and the token used.
    Must be a valid model name that can be called by using Gensim API.
    
    - top_feature:
    An integer number of most related features compare with the target word.
    
    - top_document:
    An integer number of most related documents compare with the target word.
    '''
    
    print('Looking for most related documents...')
    print(f'Using Model: {model_name}\n')
    
    model = api.load(model_name)
    
    # Check all target words are in the model
    for target_word in target_word_list:
        if target_word not in model:
            ValueError(f'Argument: target_word is not in model {model_name}')
            
    # Check input top_feature and token_list size
    if len(token_list) <= top_feature:
        print(f'Warning: Argument top_feature ({top_feature}) is greater than or equal to token_list size ({len(token_list)})')
        print('No feature filtering will be performed.\n')
        
    # Check the proportion of token that is not in model.
    token_not_exist = 0
    for i in token_list:
        if i not in model:
            token_not_exist += 1
            
    print(f'Token not in model: {token_not_exist}/{len(token_list)} ({round(token_not_exist/(len(token_list))*100,1)}%)')
    

    # Create a result dictionary for all target words
    result_dict = {}
    
    for i in range(0,len(target_word_list)):
        
        target_word = target_word_list[i]
        print(f'Comparing target word {i+1}/{len(target_word_list)}: {target_word}')
        
        token_score_df = get_target_word_similarity(model,token_list,target_word)
        doc_to_keep = get_most_similar_document(X,token_score_df,token_list,top_feature=top_feature,top_document=top_document)
        result_dict[target_word] = doc_to_keep
        
    print('\n')
    
    return result_dict

        
def get_target_word_similarity(model,token_list,target_word):
    '''
    Return the similarity score for all tokens used as feature compare with the target word.
    Using cosine similarity by default.
    '''
    
    target_word_feature = model[target_word]
    target_word_norm = np.linalg.norm(target_word_feature)
    
    # List of tokne used to build feature
    # token_list = list(vectorizer.get_feature_names_out())
    
    token_score_list = []
    
    # token_exist = 0
    # token_not_exist = 0
    
    # Use cosine similarity between the target word and the token using feature created from gensim model.
    for token in token_list:
        if token in model:
            token_norm = np.linalg.norm(model[token])
            score = (model[token] @ target_word_feature.T)/(token_norm*target_word_norm)
            # token_exist += 1
        else:
            score = None
            # token_not_exist += 1
            
        token_score_list.append(score)

    # print(f'Number of token exists in model:{token_exist}/{token_exist+token_not_exist} ({round(token_exist/(token_exist+token_not_exist)*100,1)}%)')

    df = pd.DataFrame({'token':token_list,'score':token_score_list})
    df = df.sort_values(by=['score'],ascending=False)
    df = df.fillna(0)
    
    return df

    
def get_most_similar_document(X,token_score_df,token_list,top_feature=20,top_document=100):
    
    '''
    - X:
    A 2D numpy array characteristic matrix with number of token as row, number of document as column.
    Prefer using count of token occurrence as feature.
    
    - token_score_df:
    A pandas dataframe with the similarity score for each token compare with the target word.
    
    - top_feature:
    An integer number of most related features compare with the target word.
    
    - top_document:
    An integer number of most related documents compare with the target word.
    
    Return a list of document number with zero based index.
    The list is truncated by the size of top_document.
    The document index is in the descending order of similarity score followed by the document index in the characteristic matrix.
    '''
    
    keep_feature = list(token_score_df.head(top_feature)['token'])
    
    # Loop through all token in characteristic matrix to find the feature to keep
    token_index_list = []
    for i in range(0,len(token_list)):
        if token_list[i] in keep_feature:
            token_index_list.append(i)
            
    # Filter to only contain the most related features
    X_modify = X[token_index_list,:]
    
    # Sum all remaining feature values to create a score for each document
    doc_score = np.sum(X_modify,axis=0)
    doc_rank = ss.rankdata(doc_score,method='min')
    
    distinct_rank = list(set(doc_rank))
    distinct_rank.sort()
    
    doc_to_keep = []

    # When multiple documents have the same rank, it may exceed the number of top document limit.
    # Put the document with the lowest rank into the list first.
    # Cap the list size with the number of top document limit.
    for j in distinct_rank:

        for i in range(0,len(doc_rank)):
            rank = doc_rank[i]

            if rank == j and len(doc_to_keep) < top_document:
                doc_to_keep.append(i)
                if len(doc_to_keep) >= top_document:
                    break

    return doc_to_keep




def find_association_rules(frq_items):
    # Collecting the inferred rules in a dataframe 
    rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x)) 
    rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x)) 
    rules = rules.sort_values(['lift', 'antecedent_len', 'consequents_len', 'confidence'], ascending=[False, False, False, False]) 
    return rules 


def draw_graph(rules, rules_to_show, topics, title):
    
    G1 = nx.DiGraph()
    color_map=[]
    strs = []
    for i in range(len(topics)):
        strs += dict(topics[i][1])

    for z in range(rules_to_show):
        rand_index = np.random.randint(0, len(rules) - 1)
        for a in rules.iloc[rand_index]['antecedents']:
            G1.add_nodes_from([a])
            G1.add_edge(a, rules.iloc[rand_index]['antecedents'], weight = 2)

        for c in rules.iloc[i]['consequents']:
            G1.add_nodes_from([c])
            G1.add_edge(rules.iloc[rand_index]['consequents'], c, weight=2)

        for node in G1:
            found_a_string = False
            for item in strs:
                if node==item:
                        found_a_string = True

                if found_a_string:
                        color_map.append('yellow')
                else:
                        color_map.append('green')       


    edges = G1.edges()
    pos = nx.spring_layout(G1, k=16, scale=1)
    plt.figure(figsize=(13,7))
    nx.draw_networkx(G1, pos, arrows=True, with_labels=True, node_color='lightblue', node_size=2500)            
    plt.title(title)
    plt.show()
    
    
def draw_catgraph(rules, rules_to_show, topics, title):
    N = 50
    x = rules["antecedents"]
    y = rules["consequents"]

    colors = np.random.rand(N)
    area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
    plt.figure(figsize=(13,7))
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.title(title)
    plt.show()