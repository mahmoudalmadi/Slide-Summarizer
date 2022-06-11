import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image

def show_image_group(slides_as_images, image_ids, k, summary = bool, encoder = None,slidedeck=None):
    """
    Given all the slides as images along with the indices of the desired slides,
    this function displays the desired images as a list
    """
    if summary:
        Transf = transforms.Compose([transforms.Resize((425, 475)), transforms.ToTensor()])
        imgs = [Transf(Image.open(slides_as_images[id])) for id in image_ids]
        grid = torchvision.utils.make_grid(imgs, nrow = 10)
        plt.figure(figsize=(38,28)); plt.axis(False)
        plt.imshow(F.to_pil_image(grid));
        plt.title("Slide Deck Summary with {0} Slides. Slide Indices: {1}. Total Slides in Deck: {2}" .format(k,image_ids,len(slides_as_images)), fontsize = 20)
        plt.savefig(r"C:\Users\Mahmo\Desktop\Slide-Summarizer\summarizer_outputs\slidedeck_summaries\slide_deck_summary_for_slidedeck_" + str(slidedeck)+ ".png", format = "png", dpi = 300)
    else:
        Transf = transforms.Compose([transforms.Resize((425, 475)), transforms.ToTensor()])
        imgs = [Transf(Image.open(slides_as_images[id])) for id in image_ids]
        grid = torchvision.utils.make_grid(imgs, nrow = 10)
        plt.figure(figsize=(38,28)); plt.axis(False)
        plt.imshow(F.to_pil_image(grid));
        plt.title("Top {0} Most Similar Based on {1}. Chosen Slides Indices: {2}" .format(k,str(encoder), sorted(image_ids)), fontsize = 20)
        plt.savefig(r"C:\Users\Mahmo\Desktop\Slide-Summarizer\summarizer_outputs\slidedeck_summaries\top_similar_by_text_based_on_" + str(encoder) +".pdf", format = "pdf")