from summarizer_infrastructure.feature_extraction.CLIP_features import clip_features
from summarizer_infrastructure.feature_extraction.extracting_text import extract_text
from summarizer_infrastructure.helper_functions.miscellaneous.createImageList import createImageList
from summarizer_infrastructure.helper_functions.output_formatting_tools.displaying_grid_of_slides import show_image_group
from summarizer_infrastructure.helper_functions.summarizer_skeleton.computing_importance_score import compute_importance
from summarizer_infrastructure.helper_functions.summarizer_skeleton.slided_selection import select_slides
from summarizer_infrastructure.helper_functions.summarizer_skeleton.text_enocding_distances import text_encoding_distances


def feature_extractor(pathName,text_tokenizer, text_model):
    """
    Inputs:
    pathname - path to pdf of slidedeck
    text_model - text model used for text encoding
    text_tokenizer - text model tokenizer

    Outputs
    text_distance_matrix - a 2D dictionary where text_distance_matrix[i][j] is the difference in the text encodings between slides i and j
    image_distance_matrix - a 2D dictionary where image_distance_matrix[i][j] is the difference in the image encodings between slides i and j
    text_similarty - list of similarity score where i^th entry is cumulative similarity of slide i with the rest of the slide in the deck
                     based on text encodings
    diagram_probs - list where i^th entry is the probability that i^th slide contains a diagram
    equations_probs - list where i^th entry is the probability that i^th slide contains an equation
    slides_as_images -  list of slides in slidedeck stored as images

    Given the path to a slide deck and the desired text encoder model this function computes all
    the needed features for the slide summarizer
    """

    #extract text from slidedeck
    corpus_per_page, text_per_page = extract_text(str(pathName))

    #extract all slides as images
    slides_as_images = createImageList(str(pathName))

    #use CLIP to compute three features
    diagram_probs, equation_probs, image_distance_matrix = clip_features(slides_as_images)
    
    #text encoding distances, similarity, & dissimilarity scores between slides
    text_distance_matrix, text_dissimilarity, text_similarity = text_encoding_distances(text_model, text_tokenizer, text_per_page)
    
    return text_distance_matrix, image_distance_matrix, text_similarity, diagram_probs, equation_probs, slides_as_images

def slide_summarizer(text_distance_matrix, text_similarity, image_distance_matrix,equation_probs, diagram_probs, slides_as_images, 
                     selection_removal_percentage, text_sim_weight, image_sim_weight,diagram_weight,equation_weight,
                     n_summary, percentage_of_slide_deck, slidedeck):
    """
    Inputs:
    text_sim_weight - text similarity weight
    image_sim_weight - image similarity weight
    diagram_weight - weight for importance of diagrams in slides
    equation_weight - weight for importance of equations in slides
    selection_removal_percentage - percentage of similar slides we remove upon selection of a slide
    n_summary - number of slides to include in the summary

    Output:
    Dataframe representing all the features that we based slide selection on
    
    This function also displays the summary in an image grid
    """

    if (image_sim_weight + text_sim_weight + equation_weight + diagram_weight != 1.00):
        raise ValueError("Sum of weights must add to 1")

    #computing importance score and returning features dataframe for reference
    importance_scores, features_df = compute_importance(text_similarity, image_distance_matrix, text_sim_weight, diagram_probs, 
                                                        equation_probs, image_sim_weight,diagram_weight,equation_weight)

    summarized_slidedeck, n_summary = select_slides(importance_scores, text_distance_matrix, image_distance_matrix, selection_removal_percentage, 
                                                    n_summary=n_summary, percentage_of_slide_deck=percentage_of_slide_deck)
    
    summarized_slidedeck_sorted = sorted(summarized_slidedeck)

    show_image_group(slides_as_images, summarized_slidedeck_sorted, n_summary, summary = 1, slidedeck=slidedeck) 

    return features_df 