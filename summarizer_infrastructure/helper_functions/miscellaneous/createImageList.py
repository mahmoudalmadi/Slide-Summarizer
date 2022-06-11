import sys
sys.path.append(r"c:\users\mahmo\anaconda3\envs\slide_summarizer_equipment\lib\site-packages")
from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)


def createImageList(pathName):
    """
    Input: pdf path name
    Output: list of slides in slidedeck as images
    """
    images = convert_from_path(pathName)

    image_list = []
    for i, image in enumerate(images):
        fname = "slide" + str(i) + ".png"
        image.save(fname, "PNG")
        image_list.append(fname)

    return image_list