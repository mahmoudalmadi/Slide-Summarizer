import numpy as np

def select_slides(importances, text_distance_matrix, image_distance_matrix, remove_percent, n_summary = None, percentage_of_slide_deck = None):
    """
    Inputs:
    importances - a list where importances[i] is the importance score of the i'th slide
    text_distance_matrix - a 2D dictionary where text_distance_matrix[i][j] is the difference in encodings between slides i and j
    image_distance_matrix - a 2D matrix where image_distance_matrix[i][j] is the difference in encodings between slides i and j
    remove_percent - the percent of "most similar" slides that will be removed from consideration each time a slide is added to the final deck
                      must be float between 0 and 1.
    
    Outputs
    slidedeck - indices of slides included in our summary
    n_summary - number of slides summarized in case the input was percentage_of_slide_deck
    """

    if (type(n_summary) == int) & (type(percentage_of_slide_deck) == float):
      raise  ValueError('Cannot input both n_summary & percentage_of_slide_deck')

    if percentage_of_slide_deck:
      n_summary = int(len(importances) * percentage_of_slide_deck)

    n_remove = int(len(importances) * remove_percent)
    #n_summary = 5 # Can be changed to be a percentage of the slide deck size
    slide_deck = []
    to_consider = list(range(len(importances)))

    # Convert importances to numpy array
    importances = np.array(importances)

    for i in range(n_summary):
        if (len(to_consider) < n_remove):
            break
        # Choose the slide with the highest importance score
        chosen_slide_idx = np.argmax(importances[to_consider])
        chosen_slide = to_consider[chosen_slide_idx]

        # remove chosen slide from consideration
        slide_deck.append(chosen_slide)
        to_consider.remove(chosen_slide)

        # remove n_remove most similar slides from consideration
        # First, get top n_remove most similar slides in terms of both text and distance
        # Store text distances from chosen slide deck as a n by 2 array, where column 1 is the slide number and column 2 is distance
        text_distances = np.array([*text_distance_matrix[chosen_slide].items()])
        texts_to_consider = np.argpartition(text_distances[:, 1][to_consider], n_remove)[:n_remove]
        closest_text_idxs = np.array(to_consider)[texts_to_consider]
        closest_texts = (text_distances[:, 0][closest_text_idxs]).astype(int)

        image_distances = np.array([*image_distance_matrix[chosen_slide].items()])
        images_to_consider = np.argpartition(image_distances[:, 1][to_consider], n_remove)[:n_remove]
        closest_image_idxs = np.array(to_consider)[images_to_consider]
        closest_images = (image_distances[:, 0][closest_image_idxs]).astype(int)

        # Then, remove the intersection of these 2 sets
        to_remove = set(closest_texts) & set(closest_images)
        # print(f"Removing slides {to_remove}")
        for slide in to_remove:
            to_consider.remove(slide)

    return slide_deck, n_summary