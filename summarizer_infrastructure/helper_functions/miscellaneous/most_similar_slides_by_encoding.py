from summarizer_infrastructure.helper_functions.output_formatting_tools.displaying_grid_of_slides import show_image_group

def top_k_similar_by_text(encoder, slide_number, slides_as_images, distance_matrix, k):
    """
    Inputs:
    encoder - String representing name of corresponding the distance_matrix values
                correspond to
    slide number - reference slide used to compare outputs of the different models
    slides_as_images - list of slides as images
    distance_matrix - matrix corresponding to pairwise distance between slide encodings
    k - number of closest slides to the reference slide by encoding euclidean distance

    Given the inputs, this function displays the top k closest slides to the reference
    slide as an image grid
    """

    ref_slide_distance_values = list(distance_matrix[slide_number].values())

    closest_k_slides = sorted(range(len(ref_slide_distance_values)), key = lambda sub: ref_slide_distance_values[sub])[:k+1]

    show_image_group(slides_as_images, sorted(closest_k_slides[1:]), k, summary = 0, encoder = encoder)