import numpy as np
import pandas as pd

def compute_importance(text_similarity, image_distances, text_sim_weight, diagram_probs, 
                      equation_probs, image_sim_weight,diagram_weight,equation_weight):
  
  #finding the image similarity scores for each slide
  image_sim = []
  for slide in range(0,len(text_similarity)):
    image_sim.append(sum(list(image_distances[slide].values())))

  #normalize all features
  img_sim_norm = np.array(image_sim)/ sum(image_sim)
  text_sim_norm = np.array(text_similarity)/ sum(text_similarity)
  equation_probs_norm = np.array(equation_probs)/ sum(equation_probs)
  diagram_probs_norm = np.array(diagram_probs)/ sum(diagram_probs)

  data_matrix = np.vstack((img_sim_norm, text_sim_norm, equation_probs_norm, diagram_probs_norm)).T

  slide_indices = []
  for slide in range(0,len(text_similarity)):
    slide_indices.append("Slide # " + str(slide+1))
  
  features_df = pd.DataFrame(data_matrix,index = slide_indices, columns = ["Image Similarty", "Text Similarity",
                                                                "Equation Probability", "Diagram Probability"])

  importance_scores = ((img_sim_norm * image_sim_weight)+ (text_sim_norm * text_similarity) +
                      (diagram_probs_norm * diagram_weight) + (equation_probs_norm * diagram_weight)).tolist()

  
  return importance_scores, features_df