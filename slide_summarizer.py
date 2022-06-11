from torch.ao.quantization import QuantStub
from transformers import *
from summarizer_infrastructure.helper_functions.summarizer_skeleton.overall_wrapper_function import feature_extractor, slide_summarizer
import sys 

#sys.path.append(r"c:\users\mahmo\anaconda3\envs\slide_summarizer_equipment\lib\site-packages")
#ys.path.append(r"c:\users\mahmo\appdata\roaming\python\python310\site-packages")

def slide_summarizer(slidedeck):

    scibert_uncased_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    scibert_uncased_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

    #extract features from slidedeck
    text_distance_matrix, image_distance_matrix, text_similarity, diagram_probs, equation_probs, slides_as_images = feature_extractor(slidedeck, scibert_uncased_tokenizer, scibert_uncased_model)

    #summarize slidedeck
    features_dataframe =  slide_summarizer(text_distance_matrix, text_similarity, image_distance_matrix, equation_probs, diagram_probs, slides_as_images,
                                        selection_removal_percentage = 0.05,   #percentage of similar slides to remove upon selecting a slide of interest
                                        text_sim_weight = 0.3, 
                                        image_sim_weight = 0.1, 
                                        diagram_weight = 0.2,
                                        equation_weight = 0.4,
                                        n_summary = 5,                       #number of slides to include in summary, must be None when using percentage_of_slide_deck
                                        percentage_of_slide_deck = None,     #percentage of slidedeck to select for summary, must be None when using n_summary
                                        slidedeck= slidedeck)


    return features_dataframe

features_df = slide_summarizer(r"C:\Users\Mahmo\Desktop\Slide-Summarizer\slidedecks\510.pdf")

