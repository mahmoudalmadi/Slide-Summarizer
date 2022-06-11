from collections import *
import clip
import numpy as np

def text_encoding_distances(model, tokenizer, text_per_page):
    """
    Inputs: 
          model - specifying which model to use for text encoding
          text_per_page - list of strings where each strings corresponds to all
                          the text in a given slide
          importances - dicitonary mapping each slide index to its corresponding
                        importance score
    """
    #tokenize texts and retrieve text encodings
    if tokenizer == clip.tokenize:
        tokenized_text = tokenizer(text_per_page)
        text_encodings = model(tokenized_text)
        text_encodings_processed = text_encodings.detach().numpy()
    else:
        tokenized_text = tokenizer(text_per_page, padding=True, return_tensors="pt") 
        text_encodings_unprocessed = model(**tokenized_text)
        text_encodings_processed = text_encodings_unprocessed.last_hidden_state.detach()

    #matrix whose i,j entry measures the distance between the encodings of slide i
    #and slide j
    text_distance_matrix = defaultdict(lambda: defaultdict(int))

    for slide1_text in range(0,len(text_per_page)):
        for slide2_text in range(0,len(text_per_page)):
            text_distance_matrix[slide1_text][slide2_text] = np.linalg.norm(text_encodings_processed[slide1_text] - text_encodings_processed[slide2_text]) 

    dissimilarity_scores = []
    similarity_scores = []
    for slide in range(0,len(text_per_page)):
        dissimilarity_scores.append(sum(list(text_distance_matrix[slide].values())))
        similarity_scores.append(1/sum(list(text_distance_matrix[slide].values())))
        
    return text_distance_matrix, dissimilarity_scores, similarity_scores