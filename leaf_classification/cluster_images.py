#!/usr/bin/env python
"""Group similar images via Structural Similarity Index Maximization, then
   append the group number to the file name.
   """

from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

img_dir = "./images/"

def compare_images(imageA, imageB, title):
    height, width = imageA.shape
    imageB = cv2.resize(imageB, (width, height), cv2.INTER_LINEAR)

    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)

    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    # show the images
    plt.show()

# load the images -- the original, the original + contrast,
# and the original + photoshop
quercus_brantil_1 = cv2.imread(img_dir + "14.jpg")
quercus_brantil_2 = cv2.imread(img_dir + "452.jpg")
quercus_rubra = cv2.imread(img_dir + "11.jpg")

# convert the images to grayscale
quercus_brantil_1 = cv2.cvtColor(quercus_brantil_1, cv2.COLOR_BGR2GRAY)
quercus_brantil_2 = cv2.cvtColor(quercus_brantil_2, cv2.COLOR_BGR2GRAY)
quercus_rubra = cv2.cvtColor(quercus_rubra, cv2.COLOR_BGR2GRAY)

# initialize the figure
fig = plt.figure("Images")
#images = ("Original", original), ("Contrast", contrast), ("Photoshopped", shopped)
images = ("14", quercus_brantil_1), ("452",
                                     quercus_brantil_2), ("11", quercus_rubra)

# loop over the images
for (i, (name, image)) in enumerate(images):
        # show the image
    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title(name)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis("off")

# show the figure
#plt.show()

# compare the images
#compare_images(quercus_brantil_1, quercus_brantil_1, "14 vs. 14")
#compare_images(quercus_brantil_1, quercus_brantil_2, "14 vs. 452")
#compare_images(quercus_brantil_1, quercus_rubra, "14 vs. 11")

train = pd.read_csv("train.csv")
#print train.head()
#print train.columns
#species = train.species.unique()
species = np.asarray(train.species.unique())
np.savetxt("species.csv", species,fmt='%s')
species = pd.read_csv("species.csv", header=None, names=["species"])
print species
#leaf_species = train['species','id']
