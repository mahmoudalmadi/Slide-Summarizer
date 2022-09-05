# Slide Summarizer

## About

Slide Summarizer is a project that started as a class project for COMP 646: Deep Learning in Vision Language at Rice University. The Slide Summarizer leverages highly sophisticated pre-trained computer vision and natural language processing models by incorporating them into an algorithm that summarizes slide decks. The goal of this project is to develop a study tool that can help students revise and study from slidedecks in a more effective and time-efficient manner since the most important slides will be chosen for them. At scale, this can help reduce a 200 slide slidedeck to less than 50 slides by choosing the most important and relevant slides.

This project is currently under development; however more information on the preliminary work and underlying research behind the current status of the project [is laid out in the following report.](https://drive.google.com/file/d/1yL5ck-3nixf83PcYcgOqw9fm7EegxfBw/view?usp=sharing)

To use the summarizer, one must upload their slidedecks to the **slidedecks** folder and then run the *slide_summarizer.py* file. Since this project uses a range of dependecies for incorporating machine learning models, it requires a set of libraries that are yet to be packaged into an environment and uploaded to this repository in a yml file that can be activated using Anaconda. That can then be used to run the slide summarizer.

## Repository Details

The following is a description of files and folders in the repository.

* *slide_summarizer.py*
    - Combines all of the code to implement the summarizer. Designed to be run from terminal with the options of inputting parameters. (under development)
* **appdev** folder: includes the following file
    - *webpage_dev.html*: an html script for a webpage for the slide summarizer (under development)
* **slidecks** folder: this folder contains the slidedecks to be summarized
* **summarizer_infrastructure** folder: includes folders with code used to develop the slide summarizer
    - **feature_extraction** folder: includes all the code used to extract and evaluate features from the slidedeck
        * *CLIP_features.py*: code for extracting specific slide deck features using the CLIP model
        * *extracting_text.py*: code for exctracting text from the slide deck
    - **helper_functions** folder: includes a variety of helper functons used to implement the summarizer
        * **miscellaneous** folder: 
            - *createImageList.py*: code used to convert slidedeck to a list of images
            - *most_similar_slides_by_encoding.py*: code used to display most similar slides by image/text encoding
        * **output_formatting_tools** folder:
            - *display_grid_slides.py*: code used to display output of top k slides chosen by summarizer
        * **summarizer_skeleton** folder: contains the files the carry out the main functions of the summarizer
            - *computing_importance_score.py*
            - *overall_wrapper_function.py*: combines all the functions of the summarizer to execute it
            - *slide_selection.py*: executes slide selection algorithm
            - *text_encoding_distances.py*: generates text encoding by slide and calculates a distance matrix for the distances between all the text encodings
                