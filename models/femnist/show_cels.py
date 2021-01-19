import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec


import os
import numpy as np
import random
from PIL import Image

root_path = os.path.dirname(os.path.dirname(os.path.abspath('')))
parent_path = os.path.join(root_path, "data", "celeba")
rawdata_path = os.path.join(parent_path, 'data', 'raw', 'img_align_celeba')

def get_metadata():
    f_identities = open(os.path.join(
        parent_path, 'data', 'raw', 'identity_CelebA.txt'), 'r')
    identities = f_identities.read().split('\n')

    f_attributes = open(os.path.join(
        parent_path, 'data', 'raw', 'list_attr_celeba.txt'), 'r')
    attributes = f_attributes.read().split('\n')

    return identities, attributes


def get_celebrities_and_images(identities):
    all_celebs = {}
    for line in identities:
        info = line.split()
        if len(info) < 2:
            continue
        image, celeb = info[0], info[1]
        if celeb not in all_celebs:
            all_celebs[celeb] = []
        all_celebs[celeb].append(image)

    good_celebs = {c: all_celebs[c] for c in all_celebs if len(all_celebs[c]) >= 5}
    return good_celebs
    
cels = get_celebrities_and_images(get_metadata()[0])
        

def show_by_cel(j):
    plt.figure(figsize=(10, 10))
    p = cels[j]
    if len(p) > 12:
        sub_cel = p[:12]
    else:
        sub_cel = p

    i = 0
#     size = 28, 28, 3  # original image size is 84, 84
    for f in sub_cel:
        file_path = os.path.join(rawdata_path, f)
        img = Image.open(file_path)
#         rgb = img.thumbnail(size)
        rgb = np.array(img).copy()
        plt.subplot(3, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(rgb /255.0)
        i += 1       

    plt.show()

    
