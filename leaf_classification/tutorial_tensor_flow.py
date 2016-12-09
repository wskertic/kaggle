
import os
import subprocess
import itertools
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tflearn
import glob

from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit

from skimage.io import imread
from skimage.transform import resize
from tensorflow.python.framework.ops import reset_default_graph

def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out
os.chdir("leaf_classification/")
image_paths = glob.glob("images/*")
#print os.getcwd()
#for files in os.listdir("./images/"):
#    if files.endswith(".jpg"):
#        image_paths[files]

print "Amount of image =", len(image_paths)

# resize images to consistent dimensions, small as possible
for i in range(10):
    image = imread(image_paths[i], as_grey=True)
    image = resize(image, output_shape=(100, 100))
    plt.imshow(image, cmap='gray')
    plt.title("name: %s \n shape:%s" % (image_paths[i], image.shape))
    #plt.show()

train = pd.read_csv('train.csv')
train.tail()

test = pd.read_csv('test.csv')
test.tail()

sample_submission = pd.read_csv('sample_submission.csv')
sample_submission.tail()
# name all columns in train, should be 3 different columns with 64 values each
print train.columns[2::63]

# try extracting and plotting columns
X = train.as_matrix(columns=train.columns[2:])
print "X.shape,", X.shape
margin = X[:, :64]
shape = X[:, 64:128]
texture = X[:, 128:]
print "margin.shape,", margin.shape
print "shape.shape,", shape.shape
print "texture.shape,", texture.shape

fig = plt.figure(figsize=(21,7))
for i in range(3):
    plt.subplot(3,3,1+i*3)
    fig.add_subplot(3,3,1+i*3)
    plt.plot(margin[i])
    if i == 0:
        plt.title('Margin', fontsize=20)
    plt.axis('off')
    plt.subplot(3,3,2+i*3)
    fig.add_subplot(3,3,2+i*3)
    plt.plot(shape[i])
    if i == 0:
        plt.title('Shape', fontsize=20)
    plt.axis('off')
    plt.subplot(3,3,3+i*3)
    fig.add_subplot(3,3,3+i*3)
    plt.plot(texture[i])
    if i == 0:
        plt.title('Texture', fontsize=20)

fig.set_tight_layout(True)
# plt.savefig('testsvg')
# fig.savefig('figsvg')
# fig.show()
# plt.tight_layout() # falling back to Agg renderer?????
# plt.show()

class load_data():
    # data_train, data_test, and le are public
    def __init__(self, train_path, test_path, image_paths, image_shape=(128, 128)):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        image_paths = image_paths
        image_shape = image_shape
        self._load(train_df, test_df, image_paths, image_shape)

    def _load(self, train_df, test_df, image_paths, image_paths):
        print "loading data ..."
        # load train.csv
        path_dict = self._path_to_dict(image_paths) # numerate image paths and make it a dict
        # merge image paths with data frame
        train_image_df = self._merge_image_df(train_df, path_dict)
        test_image_df = self._merge_image_df(test_df, path_dict)
        # label encoder-decoder (self. because we need it later)
        self.le = LabelEncoder().fit(train_image_df['species'])
        # labels for train
        t_train = self.le.transform(train_image_df['species'])
        # getting data
        train_data = self._make_dataset(train_image_df, image_shape, t_train)
        test_data = self._make_dataset(test_image_df, image_shape)
        # need to reformat the train for validation split reasons in the batch_generator
        self.train = self._format_dataset(train_data, for_train=True)
        self.test = self._format_dataset(test_data, for_train=False)
        print "data loaded"
