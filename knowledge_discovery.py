import pandas as pd
from transformers import pipeline
from association_analysis import *



def get_document_zero_shot_classifiaction(
    X,
    df,
    token_list,
    topic_df,
    zero_shot_model='facebook/bart-large-mnli',
    related_doc_model = 'glove-wiki-gigaword-100',
    top_feature = 20,
    top_document = 100
    ):
    
    '''
    Filter the most related documents by topic.
    Use zero-shot classification to give a label to the topic.
    '''
    
    topic_list = list(topic_df['topic'])
    label_list = list(topic_df['labels'])
    
    # Get all documents that is most related to the topics in topic_list
    index_dict = get_target_document_index(X,token_list,topic_list,model_name=related_doc_model,top_feature=top_feature,top_document=top_document)

    # Use zero-shot classification for binary classification into any 2 labels
    print('Zero shot classification in progress...')
    print(f'Using Model: {zero_shot_model}')
    
    classifier = pipeline('zero-shot-classification',model=zero_shot_model)
    
    insight_df = pd.DataFrame()
    
    # Loop through all topics to have a zero-shot classification on topic labels
    for i in range(0,len(topic_df)):
        
        topic = topic_list[i]
        labels = label_list[i]
        
        print(f"Classification on topic: '{topic}' into {labels} ({i+1}/{len(topic_df)})")

        sub_df = df[df['doc_num'].isin(index_dict[topic])]
        sub_df = sub_df[['doc_num','abstract_summary']].reset_index(drop=True)
        text_list = list(sub_df['abstract_summary'])
        
        sub_df['topic'] = topic
        
        results = classifier(text_list,candidate_labels=labels)

        # Turning the two lists into a dict for each result
        list_of_dicts = [dict(zip(x['labels'], x['scores'])) for x in results]

        result_df = pd.DataFrame(list_of_dicts).reset_index(drop=True)

        sub_df = pd.concat([sub_df,result_df],axis=1)

        # Choose the column with maximum score as the label of this document
        sub_df['label'] = sub_df[labels].idxmax(axis=1)
        
        # Get the score from the columns name as the value in the label field
        for label in labels:
            sub_df_label = sub_df[sub_df['label']==label]
            sub_df_label['score'] = sub_df_label[label]
            # sub_df_label['score'] = sub_df_label.loc[sub_df_label['label'] == label, label]
        
            sub_df_label = sub_df_label[['doc_num','topic','label','score']]
        
            # A doc_num can cover multiple topics, the unique key in insight_df is
            # the combination of doc_num and topic.
            insight_df = pd.concat([insight_df,sub_df_label])
        
    print('\n')
    
    return insight_df