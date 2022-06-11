from collections import *
import PyPDF4
import regex as re

def extract_text(pathName):
    """
    Input: pdf path name
    Output: corpus_per_page - 2D Dictionary mapping each slide to all the word
            counts for each word in the corresponding slide       
            text_per_page - list of strings where each string corresponds to all
                            the text in a given slide
    """

    pdfFileObj = open(str(pathName), 'rb')
    pdfReader = PyPDF4.PdfFileReader(pdfFileObj)

    #number of slides in slidedeck
    num_of_pages = pdfReader.numPages 

    #initializing word count dictionaries
    pdf_corpus = defaultdict(int) #keeping track of word counts accross entire pdf
    corpus_per_page = defaultdict(lambda: defaultdict(int)) #keeping track of word counts per page
    text_per_page = [] #list where each element is a string containing all the text in a given page

    for page_no in  range(0,num_of_pages):
        
        #extracting text from specific page
        pageObj = pdfReader.getPage(page_no)
        page_text = pageObj.extractText()

        filtered_text = re.split('[^a-zA-Z]', page_text) #removing anything but english words
        polished_text = [i for i in filtered_text if (i and len(i) > 3)] #removing empty strings and strings shorter than a specific length
        
        #gathering all the text in the slide into one string
        current_page = ""
        for word in polished_text:
            current_page = current_page + word + " "

        text_per_page.append(current_page)
        
        #counting the words per slide
        for word in polished_text:
            if word != " " or word != '':
                pdf_corpus[word] = pdf_corpus[word] + 1
                corpus_per_page[page_no][word] = corpus_per_page[page_no][word] + 1

    pdfFileObj.close() 

    return corpus_per_page, text_per_page