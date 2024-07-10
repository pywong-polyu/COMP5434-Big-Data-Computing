import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import random
from scipy.stats import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import hashlib

from matplotlib import pyplot as plt
# from datasketch import MinHash, Simhash

import warnings
warnings.filterwarnings('ignore')


def check_document_similarity(X,df,target_doc_num):
    '''
    Given the target document number, return the similarity to other documents.
    Output is a pandas dataframe with number of document as number of row.
    '''
    
    target_doc = X.T[target_doc_num]
    target_doc_norm = np.linalg.norm(target_doc)
    
    doc_l2_norm_list = []
    doc_l1_norm_list = []
    doc_cosine_list = []
    
    # Similarity from characteristic matrix X
    # X is 4096(feature) x 8041(document)
    
    for doc in X.T:
        doc_norm = np.linalg.norm(doc)
        doc_l2_norm = np.linalg.norm(target_doc - doc)
        doc_l1_norm = np.linalg.norm(target_doc - doc,ord=1)
        doc_cosine = doc@target_doc.T/(doc_norm*target_doc_norm)

        doc_l2_norm_list.append(doc_l2_norm)
        doc_l1_norm_list.append(doc_l1_norm)
        doc_cosine_list.append(doc_cosine)

    
    result = pd.DataFrame(
        {
            'doc_l1_norm':doc_l1_norm_list,
            'doc_l2_norm':doc_l2_norm_list,
            'doc_cosine_similarity':doc_cosine_list,
            # 'sig_cosine_similarity':sig_cosine_list,
            # 'sig_jaccard_similarity':sig_jaccard_list
        }
    )
    
    result = result.reset_index().rename(columns={'index':'doc_num'})
    result = result.merge(df[['doc_num','abstract_summary','doc_type']],how='left',on='doc_num')
    
    return result


def get_permutate_matrix(X,num_perm=50):
    
    '''
    Return premutation matrix with size:
    number of feature x number of permutation
    '''
    
    N = len(X) # number of features
    p = []

    for i in range(0,num_perm):
        a = random.randint(0, N-1)
        b = random.randint(0, N-1)
        
        new_order = np.array([])
        not_used = list(range(1,N+1))
        
        for j in range(0,N):
            # Buckets are set to be 1 to N
            hash_value = (a*j+b)%N+1
            # When 2 hashes mapped to the same bucket, the second hash move to the next bucket.
            while hash_value not in not_used:
                hash_value += 1
                if hash_value > N:
                    hash_value = 1

            not_used.remove(hash_value)
            new_order = np.append(new_order,hash_value)
              
        p.append(new_order)
        
    p = np.array(p).T
    
    return p

def get_minhash_signature(X,p):
    '''
    Return signature matrix with size: number of permutation x number of document
    '''

    s = []

    for perm in p.T:
        doc_sig = np.array([])
        for doc in X.T:
            sig_set = set(np.multiply(doc,perm))
            sig_set.discard(0)
            sig = min(sig_set)
            doc_sig = np.append(doc_sig,sig)
            
        s.append(doc_sig)

    s = np.array(s)
    
    return s

def check_minhash_similarity(s,X,target_doc_num):

    sig_jaccard_list = []
    sig_cosine_list = []
    doc_jaccard_list = []
    
    # Similarity from signature matrix
    # s is (permutation) x (document)
    
    target_sig = s.T[target_doc_num]
    target_sig_norm = np.linalg.norm(target_sig)
    
    target_doc = X.T[target_doc_num]
    
    # loop through all document to create a list of score
    for sig in s.T:
        sig_norm = np.linalg.norm(sig)
        sig_cosine = (sig @ target_sig.T)/(sig_norm*target_sig_norm)
        sig_cosine_list.append(sig_cosine)

        sig_intersection = len(list(set(target_sig).intersection(sig)))
        sig_union = (len(set(target_sig)) + len(set(sig))) - sig_intersection
        sig_jaccard = sig_intersection / sig_union
        sig_jaccard_list.append(sig_jaccard)   

    for doc in X.T:
        doc_intersection = target_doc @ doc.T
        doc_union = sum(np.clip(target_doc+doc,0,1))
        doc_jaccard = doc_intersection / doc_union
        doc_jaccard_list.append(doc_jaccard)
        
        
    df = pd.DataFrame(
        {
            'sig_cosine_similarity':sig_cosine_list,
            'sig_jaccard_similarity':sig_jaccard_list,
            'doc_jaccard_similarity':doc_jaccard_list
        }
    )
    
    df = df.reset_index().rename(columns={'index':'doc_num'})
    
    return df

def minhash_with_different_permutation(df,X_clip,clip_result,target_doc_num,num_perm_list):
    
    '''
    The only parameter to fine tune in MinHash is the number of permutation used to generate the signature matrix.
    '''
    
    permutation_dict = {}
    
    clip_result = check_document_similarity(X_clip,df,target_doc_num)
    
    for num_perm in num_perm_list:
        p = get_permutate_matrix(X_clip,num_perm=num_perm)
        s = get_minhash_signature(X_clip,p)
        
        minhash_result = check_minhash_similarity(s,X_clip,target_doc_num)
        minhash_result = clip_result.merge(minhash_result,how='outer',on='doc_num')
        minhash_result = minhash_result.sort_values(by=['sig_jaccard_similarity'],ascending=False)
        
        permutation_dict[num_perm] = minhash_result
    
    return permutation_dict


def compare_minhash_fine_tune(permutation_dict):
    
    '''
    The performance of MinHash is defined by how the signature matrix gives a 
    close similarity score compare with the characteristic matrix.
    In the other words, documents have a high similarity score to a target 
    document will have a close similarity score when using the signature matrix.
    The relation between the characteristic matrix similarity and the signature 
    matrix similarity can be expressed as the correlation between the characteristic 
    matrix Jaccard similarity and the signature matrix Jaccard similarity.
    Therefore, a higher Pearson correlation implies a better performance in 
    dimension reduction while maintaining the data pattern in the characteristic matrix.
    '''
    
    num_perm_list = []
    correlation_list = []
    pvalue_list = []
    
    for num_perm in permutation_dict:
        
        minhash_result = permutation_dict[num_perm]
        correlation, pvalue = pearsonr(minhash_result['sig_jaccard_similarity'],minhash_result['doc_jaccard_similarity'])
        
        num_perm_list.append(num_perm)
        correlation_list.append(correlation)
        pvalue_list.append(pvalue)
        
    df = pd.DataFrame({
        'num_perm':num_perm_list,
        'correlation':correlation_list,
        'pvalue':pvalue_list
    })
    
    df = df.sort_values(by=['num_perm'],ascending=False)
    
    return df

def get_feature_hash(feature_list,hashing):
    
    if hashing not in ['md5','sha1']:
        raise ValueError(f'Argument hashing is not md5 or sha1: {hashing}')
        
    feature_hash = []
    max_bit_size = 0
    
    for feature_name in feature_list:        
        # Use MD5 or SHA1 to encode
        # Convert the integer into string of 0 or 1 as a hash.
        
        encod_feature = str.encode(feature_name)
        
        # MD5
        if hashing == 'md5':
            feature_byte = hashlib.md5(encod_feature).digest()
            feature_int = int.from_bytes(feature_byte, byteorder='little')
        
        # SHA1
        elif hashing == 'sha1':
            feature_byte = hashlib.sha1(encod_feature).hexdigest()
            feature_int = int(feature_byte, 16)
        
        feature_bit = f'{feature_int:0b}'
        feature_hash.append(feature_bit)
        
        bit_size = feature_int.bit_length()
        
        if bit_size > max_bit_size:
            max_bit_size = bit_size
    
    # Convert a string of bits into np.array
    for i in range(0,len(feature_hash)):
        # If the integer has less bits, make sure it has 128bits
        feature_hash[i] = feature_hash[i].zfill(max_bit_size)
        
        # Convert a string of bits into np.array
        bit = np.array([])
        for j in feature_hash[i]:
            bit = np.append(bit,int(j))
        feature_hash[i] = bit
       
    feature_hash = np.array(feature_hash)
    
    return feature_hash


def get_doc_finger_print(feature_hash,doc):
    
    total_bit = len(feature_hash[0])
    finger_print = np.zeros(total_bit)
    
    # Loop through each feature
    for i in range(0,len(doc)):
        feature_value = doc[i]
        hash = feature_hash[i] # hash of the ith feature
        
        feature_weight = (hash*2-1)*feature_value
        finger_print += feature_weight
    
    # Loop through each bit in hash
    for bit_pos in range(0,total_bit):
        # Determine the finger print bit by the bit_weight
        if finger_print[bit_pos] > 0:
            finger_print[bit_pos] = 1
        elif finger_print[bit_pos] < 0:
            finger_print[bit_pos] = 0
        # When the bit has same number of feature as 1 and 0, use the bit position to determine the finger print bit.
        else:
            finger_print[bit_pos] = bit_pos%2     
        
    return finger_print


def get_finger_print_list(X,feature_hash):
    
    finger_print_list = []
    
    for doc in X.T:
        finger_print = get_doc_finger_print(feature_hash,doc)
        finger_print_list.append(finger_print)
    
    return finger_print_list


def check_simhash_similarity(finger_print_list,target_doc_num):
    
    # sig_cosine_list = []
    sig_hamming_distance_list = []
    
    target_sig = finger_print_list[target_doc_num]
    # target_sig_norm = np.linalg.norm(target_sig)
    
    for sig in finger_print_list:
        # sig_norm = np.linalg.norm(sig)
        # sig_cosine = (sig @ target_sig.T)/(sig_norm*target_sig_norm)
        # sig_cosine_list.append(sig_cosine)
        
        # Distance is the number of different bit
        sig_hamming_distance = 0
        
        for i in range(0,len(sig)):
            if sig[i] != target_sig[i]:
                sig_hamming_distance += 1
        
        sig_hamming_distance_list.append(sig_hamming_distance)
    
    df = pd.DataFrame(
        {
            # 'sig_cosine_similarity':sig_cosine_list,
            'sig_hamming_distance':sig_hamming_distance_list
        }
    )
    
    df = df.reset_index().rename(columns={'index':'doc_num'})
    
    return df


def tag_documents(tokenizer,orginal_docs):
    tagged_docs = []
    for i in range(0,len(orginal_docs)):
        doc = orginal_docs[i]
        doc_id = str(i)
        tokens = tokenizer.tokenize(doc)
        tagged_doc = TaggedDocument(words=tokens,tags=[doc_id])
        tagged_docs.append(tagged_doc)
        
    return tagged_docs


def get_doc2vec_result(model,tokenizer,original_docs,target_doc_num,df):
    
    target_doc = original_docs[target_doc_num]
    target_tokens = tokenizer.tokenize(target_doc)
    inferred_vector = model.infer_vector(target_tokens)
    similar_docs = model.dv.most_similar([inferred_vector],topn=len(original_docs))
    
        
    doc_num_list = []
    # rank_list = []
    similarity_list = []

    for i in range(0,len(similar_docs)):
        doc = similar_docs[i]
        # rank start from 1 instead of 0, 1 has highest similarity
        # rank = i + 1
        doc_num = int(doc[0])
        similarity = doc[1]
        
        # rank_list.append(rank)
        similarity_list.append(similarity)
        doc_num_list.append(doc_num)
        
    result = pd.DataFrame({'doc_num':doc_num_list,'doc2vec_similarity':similarity_list})
    result = result.merge(df[['doc_num','abstract_summary','doc_type']],how='left',on='doc_num')
    
    return result

def get_combined_similarity_result(tfidf_result,count_result,clip_result,minhash_result,simhash_result,doc2vec_result):
    
    tfidf_result = tfidf_result[['doc_num','doc_cosine_similarity']]
    tfidf_result = tfidf_result.rename(columns={'doc_cosine_similarity':'tfidf_doc_cosine'})
    
    count_result = count_result[['doc_num','doc_l2_norm']]
    count_result = count_result.rename(columns={'doc_l2_norm':'count_doc_l2_norm'})
    
    clip_result = clip_result[['doc_num','doc_l1_norm']]
    clip_result = clip_result.rename(columns={'doc_l1_norm':'clip_doc_l1_norm'})
    
    minhash_result = minhash_result[['doc_num','sig_jaccard_similarity']]
    minhash_result = minhash_result.rename(columns={'sig_jaccard_similarity':'minhash_sig_jaccard'})
    
    simhash_result = simhash_result[['doc_num','sig_hamming_distance']]
    simhash_result = simhash_result.rename(columns={'sig_hamming_distance':'simhash_sig_hamming'})
    
    
    combined_result = tfidf_result.merge(count_result,how='outer',on='doc_num')
    combined_result = combined_result.merge(clip_result,how='outer',on='doc_num')
    combined_result = combined_result.merge(minhash_result,how='outer',on='doc_num')
    combined_result = combined_result.merge(simhash_result,how='outer',on='doc_num')
    combined_result = combined_result.merge(doc2vec_result,how='outer',on='doc_num')
    
    return combined_result
    
def get_performance_rank(tfidf_result,count_result,clip_result,minhash_result,simhash_result,doc2vec_result):

    # Create ranking of similarity from previous result, similar item has the lowest rank

    # Rank from directly comparing the document features
    tfidf_rank = tfidf_result[['doc_num','doc_cosine_similarity']]
    tfidf_rank['tfidf_doc_cosine'] = tfidf_rank['doc_cosine_similarity'].rank(ascending=False)
    # tfidf_rank['tfidf_doc_l1_norm'] = tfidf_rank['doc_l1_norm'].rank(ascending=True)
    # tfidf_rank['tfidf_doc_l2_norm'] = tfidf_rank['doc_l2_norm'].rank(ascending=True)
    performance_df = tfidf_rank[['doc_num','tfidf_doc_cosine']]

    count_rank = count_result[['doc_num','doc_l2_norm']]
    # count_rank['count_doc_cosine'] = count_rank['doc_cosine_similarity'].rank(ascending=False)
    # count_rank['count_doc_l1_norm'] = count_rank['doc_l1_norm'].rank(ascending=True)
    count_rank['count_doc_l2_norm'] = count_rank['doc_l2_norm'].rank(ascending=True)
    performance_df = performance_df.merge(count_rank[['doc_num','count_doc_l2_norm']],how='outer',on='doc_num')

    clip_rank = clip_result[['doc_num','doc_l1_norm']]
    # clip_rank['clip_doc_cosine'] = clip_rank['doc_cosine_similarity'].rank(ascending=False)
    clip_rank['clip_doc_l1_norm'] = clip_rank['doc_l1_norm'].rank(ascending=True)
    # clip_rank['clip_doc_l2_norm'] = clip_rank['doc_l2_norm'].rank(ascending=True)
    performance_df = performance_df.merge(clip_rank[['doc_num','clip_doc_l1_norm']],how='outer',on='doc_num')


    # Rank from comparing the document features
    minhash_rank = minhash_result[['doc_num','sig_cosine_similarity','sig_jaccard_similarity']]
    # minhash_rank['minhash_cosine'] = minhash_rank['sig_cosine_similarity'].rank(ascending=False)
    minhash_rank['minhash_jaccard'] = minhash_rank['sig_jaccard_similarity'].rank(ascending=False)
    performance_df = performance_df.merge(minhash_rank[['doc_num','minhash_jaccard']],how='outer',on='doc_num')

    simhash_rank = simhash_result[['doc_num','sig_hamming_distance']]
    simhash_rank['simhash_hamming'] = simhash_rank['sig_hamming_distance'].rank(ascending=True)
    performance_df = performance_df.merge(simhash_rank[['doc_num','simhash_hamming']],how='outer',on='doc_num')
    
    doc2vec_rank = doc2vec_result[['doc_num','doc2vec_similarity']]
    doc2vec_rank['doc2vec'] = doc2vec_rank['doc2vec_similarity'].rank(ascending=False)
    performance_df = performance_df.merge(doc2vec_rank[['doc_num','doc2vec']],how='outer',on='doc_num')
    
    return performance_df


def get_performance_rank_diff(df,benchmark='tfidf_doc_cosine'):
    '''
    Compare the performance with the benchmark field.
    By default it is the cosine similarity of document TF-IDF.
    '''
            
    for col in df:
        if col not in ['doc_num',benchmark]:
            df[col] = df[col] - df[benchmark]
    
    return df