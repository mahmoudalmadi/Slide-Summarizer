from collections import *
import torch
import clip
from PIL import Image
import numpy as np

def clip_features(slides_in_deck):
    """
    Inputs:
    slides_in_deck - list of slides in slidedeck as images

    Outputs:
    diagram_probs - list where i^th entry is the probability that i^th slide contains a diagram
    equation_probs - list where i^th entry is the probability that i^th slide contains an equation
    image_distance_matrix - matrix where entry (i, j) is distance between slides i and j as images

    Identifiying objects in the slides with clip and returning CLIP image  encodings
    """

    importances = defaultdict(lambda:1)

    # Loading CLIP Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load('ViT-B/32', device)

    # Prepare text inputs
    classes = ["diagram", "chart", "figure", "equation"]
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
    with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    all_image_features = []
    diagram_probs = []
    equation_probs = []
    index = 0
    for slide in list(slides_in_deck):
        image_input = preprocess(Image.open(slide)).unsqueeze(0).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)

        # Pick the most similar label for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        all_image_features.append(image_features)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        diagram_probs.append(similarity[0][0].item())
        equation_probs.append(similarity[0][3].item())

    #matrix whose i,j entry measures the distance between the encodings of slide i
    #and slide j
    image_distance_matrix = defaultdict(lambda: defaultdict(int))

    for slide1 in range(0,len(slides_in_deck)):
        for slide2 in range(0,len(slides_in_deck)):
            image_distance_matrix[slide1][slide2] = np.linalg.norm(all_image_features[slide1] - all_image_features[slide2])

    return (diagram_probs, equation_probs, image_distance_matrix)