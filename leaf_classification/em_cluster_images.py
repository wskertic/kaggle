#!/usr/bin/env python
"""Group similar images via Structural Similarity Index Maximization, then
   append the group number to the file name.
   """

import os, glob
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
import matplotlib.pyplot as plt
import numpy as np
import cv2

img_dir = "./images/"

em = cv2.ml.EM_create()
em.setClustersNumber(99)

leaves = []
for infile in glob.glob( os.path.join(img_dir, '*.jpg') ):
    leaves.append(cv2.imread(infile, 0))
leaves[15]
#np.asarray(leaves[:10])
print type(leaves)
# initialize the figure
fig = plt.figure("Leaves")
# loop over the images
for (i, image) in enumerate(leaves):
        # show the image
    ax = fig.add_subplot(1, len(leaves), i + 1)
    #ax.set_title(image)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis("off")

# show the figure
#plt.show()
#em.trainEM(leaves)
