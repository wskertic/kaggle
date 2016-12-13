"""Walks through TFLearn tutorial for Kaggle Titnanic."""
# import os
import numpy as np
import pandas as pd
import tflearn
# import tensorflow as tf

# from tflearn.datasets import titanic
# titanic.download_dataset('titanic_dataset.csv')

from tflearn.data_utils import load_csv
# data, labels = load_csv('titanic_dataset.csv', target_column=0,
#                         categorical_labels=True, n_classes=2)
data, labels = load_csv('data/train.csv', target_column=1,
                        categorical_labels=True, n_classes=2, columns_to_ignore=[0, 10, 11])
# print(data)
# Preprocessing function


def preprocess(data, columns_to_ignore):
    # Sort by descending id and delete columns
    for id in sorted(columns_to_ignore, reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
        # Converting 'sex' field to float (id is 1 after removing Labels
        # column)
        data[i][1] = 1. if data[i][1] == 'female' else 0.
        if data[i][2] == '':
            data[i][2] = 0.
    # print(data)
    return np.array(data, dtype=np.float32)

# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
to_ignore = [1, 6]

# Preprocess data
data = preprocess(data, to_ignore)
# print(data.shape)
# Build neural network
net = tflearn.input_data(shape=[None, 6])
# net = tflearn.input_data(shape=[None, 8])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)
# net = tflearn.regression(net, to_one_hot=True, n_classes=2)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

# Let's create some data for DiCaprio and Winslet
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]

# Preprocess data
# dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
# Predict surviving chances (class 1 results)
# pred = model.predict([dicaprio, winslet])
# print("DiCaprio Surviving Rate:", pred[0][1])
# print("Winslet Surviving Rate:", pred[1][1])

# import pandas as pd
# import numpy as np
test = pd.read_csv('data/test.csv', usecols=[0, 1, 3, 4, 5, 6, 8], index_col=0)
test['Sex'] = test['Sex'].map({'female': 1., 'male': 0.})
test.replace(to_replace=np.nan, value=0., inplace=True)

people = test.T.to_dict('list')
passengers = [k for k in people]
pass_data = [v for v in people.values()]
# print(len(pass_data), len(passengers))
pred = model.predict(pass_data)

survivors = []
for survivor in range(len(pred)):
    survivors.append(1 if pred[survivor][1] > pred[survivor][0] else 0)

df = pd.DataFrame(survivors, index=passengers)

print df.head()
df.to_csv('tfl_tit.csv', header=False)


# for each in range(len(pred)):
# print "%s," % passengers[each], "1" if pred[each][1] > pred[each][0]
# else "0"
