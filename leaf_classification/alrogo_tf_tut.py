%matplotlib inline
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
print "Amount of images =", len(image_paths)
# now plot 10 images
# as we need all images to have the same dimensionality, we will resize and plot
# make the images as small as possible, until the difference between starts to get blurry
for i in range(10):
    image = imread(image_paths[i], as_grey=True)
    #image = resize(image, output_shape=(100, 100))
    plt.imshow(image, cmap='gray')
    plt.title("name: %s \n shape:%s" % (image_paths[i], image.shape))
    plt.show()

# now loading the train.csv to find features for each training point
train = pd.read_csv('train.csv')
# notice how we "only" have 990 (989+0 elem) images for training, the rest is for testing
train.tail()

# now do similar as in train example above for test.csv
test = pd.read_csv('test.csv')
# notice that we do not have species here, we need to predict that ..!
test.tail()
# and now do similar as in train example above for test.csv
sample_submission = pd.read_csv('sample_submission.csv')
# accordingly to these IDs we need to provide the probability of a given plant being present
sample_submission.tail()

# name all columns in train, should be 3 different columns with 64 values each
print train.columns[2::64]
# try and extract and plot columns
X = train.as_matrix(columns=train.columns[2:])
print "X.shape,", X.shape
margin = X[:, :64]
shape = X[:, 64:128]
texture = X[:, 128:]
print "margin.shape,", margin.shape
print "shape.shape,", shape.shape
print "texture.shape,", texture.shape
# let us plot some of the features
plt.figure(figsize=(21,7))
for i in range(3):
    plt.subplot(3,3,1+i*3)
    plt.plot(margin[i])
    if i == 0:
        plt.title('Margin', fontsize=20)
    plt.axis('off')
    plt.subplot(3,3,2+i*3)
    plt.plot(shape[i])
    if i == 0:
        plt.title('Shape', fontsize=20)
    plt.axis('off')
    plt.subplot(3,3,3+i*3)
    plt.plot(texture[i])
    if i == 0:
        plt.title('Texture', fontsize=20)

plt.tight_layout()
plt.show()

class load_data():
    # data_train, data_test and le are public
    def __init__(self, train_path, test_path, image_paths, image_shape=(128, 128)):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        image_paths = image_paths
        image_shape = image_shape
        self._load(train_df, test_df, image_paths, image_shape)

    def _load(self, train_df, test_df, image_paths, image_shape):
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


    def _path_to_dict(self, image_paths):
        path_dict = dict()
        for image_path in image_paths:
            num_path = int(os.path.basename(image_path[:-4]))
            path_dict[num_path] = image_path
        return path_dict

    def _merge_image_df(self, df, path_dict):
        split_path_dict = dict()
        for index, row in df.iterrows():
            split_path_dict[row['id']] = path_dict[row['id']]
        image_frame = pd.DataFrame(split_path_dict.values(), columns=['image'])
        df_image =  pd.concat([image_frame, df], axis=1)
        return df_image


    def _make_dataset(self, df, image_shape, t_train=None):
        if t_train is not None:
            print "loading train ..."
        else:
            print "loading test ..."
        # make dataset
        data = dict()
        # merge image with 3x64 features
        for i, dat in enumerate(df.iterrows()):
            index, row = dat
            sample = dict()
            if t_train is not None:
                features = row.drop(['id', 'species', 'image'], axis=0).values
            else:
                features = row.drop(['id', 'image'], axis=0).values
            sample['margin'] = features[:64]
            sample['shape'] = features[64:128]
            sample['texture'] = features[128:]
            if t_train is not None:
                sample['t'] = np.asarray(t_train[i], dtype='int32')
            image = imread(row['image'], as_grey=True)
            image = resize(image, output_shape=image_shape)
            image = np.expand_dims(image, axis=2)
            sample['image'] = image
            data[row['id']] = sample
            if i % 100 == 0:
                print "\t%d of %d" % (i, len(df))
        return data

    def _format_dataset(self, df, for_train):
        # making arrays with all data in, is nessesary when doing validation split
        data = dict()
        value = df.values()[0]
        img_tot_shp = tuple([len(df)] + list(value['image'].shape))
        data['images'] = np.zeros(img_tot_shp, dtype='float32')
        feature_tot_shp = (len(df), 64)
        data['margins'] = np.zeros(feature_tot_shp, dtype='float32')
        data['shapes'] = np.zeros(feature_tot_shp, dtype='float32')
        data['textures'] = np.zeros(feature_tot_shp, dtype='float32')
        if for_train:
            data['ts'] = np.zeros((len(df),), dtype='int32')
        else:
            data['ids'] = np.zeros((len(df),), dtype='int32')
        for i, pair in enumerate(df.items()):
            key, value = pair
            data['images'][i] = value['image']
            data['margins'][i] = value['margin']
            data['shapes'][i] = value['shape']
            data['textures'][i] = value['texture']
            if for_train:
                data['ts'][i] = value['t']
            else:
                data['ids'][i] = key
        return data

# loading data and setting up constants
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
IMAGE_PATHS = glob.glob("images/*.jpg")
NUM_CLASSES = 99
IMAGE_SHAPE = (128, 128, 1)
NUM_FEATURES = 64 # for all three features, margin, shape and texture
# train holds both X (input) and t (target/truth)
data = load_data(train_path=TRAIN_PATH, test_path=TEST_PATH,
                 image_paths=IMAGE_PATHS, image_shape=IMAGE_SHAPE[:2])
# to visualize the size of the dimensions of the data
print
print "@@@Shape checking of data sets@@@"
print
print "TRAIN"
print "\timages\t%s%f" % (data.train['images'].shape, data.train['images'].mean())
print "\tmargins\t%s\t%f" % (data.train['margins'].shape, data.train['margins'].mean())
print "\tshapes\t%s\t%f" % (data.train['shapes'].shape, data.train['shapes'].mean())
print "\ttextures%s\t%f" % (data.train['textures'].shape, data.train['textures'].mean())
print "\tts\t %s" % (data.train['ts'].shape)
print "\twhile training, batch_generator will onehot encode ts to (batch_size, num_classes)"
print
print "TEST"
print "\timages\t%s\t%f" % (data.test['images'].shape, data.test['images'].mean())
print "\tmargins\t%s\t%f" % (data.test['margins'].shape, data.test['margins'].mean())
print "\tshapes\t%s\t%f" % (data.test['shapes'].shape, data.test['shapes'].mean())
print "\ttextures%s\t%f" % (data.test['textures'].shape, data.test['textures'].mean())
print "\tids\t%s" % (data.test['ids'].shape)

class batch_generator():
    def __init__(self, data, batch_size=64, num_classes=99,
                 num_iterations=5e3, num_features=64, seed=42, val_size=0.1):
        print "initiating batch generator"
        self._train = data.train
        self._test = data.test
        # get image size
        value = self._train['images'][0]
        self._image_shape = list(value.shape)
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._num_iterations = num_iterations
        self._num_features = num_features
        self._seed = seed
        self._val_size = 0.1
        self._valid_split()
        print "batch generator initiated ..."

    def _valid_split(self):
        self._idcs_train, self._idcs_valid = iter(
            StratifiedShuffleSplit(self._train['ts'],
                                   n_iter=1,
                                   test_size=self._val_size,
                                   random_state=self._seed)).next()
    def _shuffle_train(self):
        np.random.shuffle(self._idcs_train)

    def _batch_init(self, purpose):
        assert purpose in ['train', 'valid', 'test']
        batch_holder = dict()
        batch_holder['margins'] = np.zeros((self._batch_size, self._num_features), dtype='float32')
        batch_holder['shapes'] = np.zeros((self._batch_size, self._num_features), dtype='float32')
        batch_holder['textures'] = np.zeros((self._batch_size, self._num_features), dtype='float32')
        batch_holder['images'] = np.zeros(tuple([self._batch_size] + self._image_shape), dtype='float32')
        if (purpose == "train") or (purpose == "valid"):
            batch_holder['ts'] = np.zeros((self._batch_size, self._num_classes), dtype='float32')
        else:
            batch_holder['ids'] = []
        return batch_holder

    def gen_valid(self):
        batch = self._batch_init(purpose='train')
        i = 0
        for idx in self._idcs_valid:
            batch['margins'][i] = self._train['margins'][idx]
            batch['shapes'][i] = self._train['shapes'][idx]
            batch['textures'][i] = self._train['textures'][idx]
            batch['images'][i] = self._train['images'][idx]
            batch['ts'][i] = onehot(np.asarray([self._train['ts'][idx]], dtype='float32'), self._num_classes)
            i += 1
            if i >= self._batch_size:
                yield batch, i
                batch = self._batch_init(purpose='valid')
                i = 0
        if i != 0:
            yield batch, i

    def gen_test(self):
        batch = self._batch_init(purpose='test')
        i = 0
        for idx in range(len(self._test['ids'])):
            batch['margins'][i] = self._test['margins'][idx]
            batch['shapes'][i] = self._test['shapes'][idx]
            batch['textures'][i] = self._test['textures'][idx]
            batch['images'][i] = self._test['images'][idx]
            batch['ids'].append(self._test['ids'][idx])
            i += 1
            if i >= self._batch_size:
                yield batch, i
                batch = self._batch_init(purpose='test')
                i = 0
        if i != 0:
            yield batch, i

    def gen_train(self):
        batch = self._batch_init(purpose='train')
        iteration = 0
        i = 0
        while True: # REALLY???
            # shuffling all batches
            self._shuffle_train()
            for idx in self._idcs_train:
                # extract data from dict
                batch['margins'][i] = self._train['margins'][idx]
                batch['shapes'][i] = self._train['shapes'][idx]
                batch['textures'][i] = self._train['textures'][idx]
                batch['images'][i] = self._train['images'][idx]
                batch['ts'][i] = onehot(np.asarray([self._train['ts'][idx]], dtype='float32'), self._num_classes)
                i += 1
                if i >= self._batch_size:
                    yield batch
                    batch = self._batch_init(purpose='train')
                    i = 0
                    iteration += 1
                    if iteration >= self._num_iterations:
                        break

dummy_batch_gen = batch_generator(data, batch_size=64, num_classes=99, num_iterations=5e3, seed=42)
train_batch = dummy_batch_gen.gen_train().next()
valid_batch, i = dummy_batch_gen.gen_valid().next()
test_batch, i = dummy_batch_gen.gen_test().next()

print "\n"
print "@@@Shape/mean checking of batches@@@"
print
print "TRAIN"
print "\timages,", train_batch['images'].shape
print "\tmargins,", train_batch['margins'].shape
print "\tshapes,", train_batch['shapes'].shape
print "\ttextures,", train_batch['textures'].shape
print "\tts,", train_batch['ts'].shape
print
print "VALID"
print "\timages,", valid_batch['images'].shape
print "\tmargins,", valid_batch['margins'].shape
print "\tshapes,", valid_batch['shapes'].shape
print "\ttextures,", valid_batch['textures'].shape
print "\tts,", valid_batch['ts'].shape
print
print "TEST"
print "\timages,", test_batch['images'].shape
print "\tmargins,", test_batch['margins'].shape
print "\tshapes,", test_batch['shapes'].shape
print "\ttextures,", test_batch['textures'].shape
print "\tids,", len(test_batch['ids'])
# notice that mean is very different, which is why we use batch_norm in all input data in model
