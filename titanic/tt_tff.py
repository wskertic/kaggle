"""Walks through tflearn tutorial for Kaggle Titnanic."""
# import os
import numpy as np
# import pandas as pd
import tflearn
# import tensorflow as tf

from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

from tflearn.data_utils import load_csv
data, labels = load_csv('titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)
# data, labels = load_csv('data/train.csv', target_column=0,
#                         categorical_labels=True, n_classes=2)

print('it worked...')
# print(data)
