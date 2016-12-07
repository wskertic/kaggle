import numpy as np
import matplotlib.pyplot as plt
import glob, os, os.path
import cv2

IMAGE_X, IMAGE_Y = 420, 580

SUBSET = 500

# In[2]:

image_dir = "/Users/Will/Documents/DataScience/Kaggle/Ultrasound Nerve Segmentation/"

f_ultrasounds = [img for img in glob.glob(image_dir+"train/*.tif") if 'mask' not in img]
images = [ plt.imread(f_ultrasound).astype(np.float32)/256. for f_ultrasound in f_ultrasounds[:SUBSET] ]


# In[3]:

f_masks = [img for img in glob.glob(image_dir+"train/*.tif") if 'mask' in img]
masks = [ cv2.imread(mfile, -1) for mfile in f_masks[:SUBSET] ]


# In[4]:
def get_ellipse(mask):
   ret, thresh = cv2.threshold(mask, 127, 255, 0)
   contours, hierarchy = cv2.findContours(thresh, 1, 2)
   has_ellipse = len(contours) > 0
   if has_ellipse:
       cnt = contours[0]
       ellipse = cv2.fitEllipse(cnt)
       return float(has_ellipse), ellipse[0][1] / IMAGE_X, ellipse[0][0] / IMAGE_Y
   else:
       return 0., 0., 0.


# In[5]:

image_targets = [ get_ellipse(mask) for mask in masks ]

# In[9]:


# In[6]:

OUTPUT_NODES = 3

import tensorflow as tf
targets = tf.placeholder("float", (None, OUTPUT_NODES))
inputs = tf.placeholder("float", (None, IMAGE_X, IMAGE_Y))
#batch_size_placeholder = tf.placeholder(tf.int32)

# network here

HIDDEN_LAYER = 256

convolution_weights_1 = tf.Variable(tf.truncated_normal([4, 4, 1, 12], stddev=0.01))
convolution_bias_1 = tf.Variable(tf.constant(0.01, shape=[12]))

convolution_weights_2 = tf.Variable(tf.truncated_normal([4, 4, 12, 12], stddev=0.01))
convolution_bias_2 = tf.Variable(tf.constant(0.01, shape=[12]))

feed_forward_weights_1 = tf.Variable(tf.truncated_normal([3192, HIDDEN_LAYER], stddev=0.01))
feed_forward_bias_1 = tf.Variable(tf.constant(0.0, shape=[HIDDEN_LAYER]))

feed_forward_weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_LAYER, OUTPUT_NODES], stddev=0.01))
feed_forward_bias_2 = tf.Variable(tf.constant(0.0, shape=[OUTPUT_NODES]))

hidden_convolutional_layer_1 = tf.nn.relu(
   tf.nn.conv2d(tf.reshape(inputs, (-1, IMAGE_X, IMAGE_Y, 1)), convolution_weights_1, strides=[1, 4, 4, 1],
                padding="SAME") + convolution_bias_1)

hidden_max_pooling_layer_1 = tf.nn.max_pool(hidden_convolutional_layer_1, ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1], padding="SAME")

hidden_max_pooling_layer_2 = tf.nn.conv2d(hidden_max_pooling_layer_1, tf.Variable(tf.truncated_normal([4, 4, 12, 12], stddev=0.01)),
            strides=[1, 4, 4, 1], padding="SAME")

hidden_activations = tf.nn.relu(
          tf.matmul(tf.reshape(hidden_max_pooling_layer_2, (-1, 3192)), feed_forward_weights_1) + feed_forward_bias_1)

output = tf.nn.sigmoid(
          tf.matmul(hidden_activations, feed_forward_weights_2) + feed_forward_bias_2)


#loss_func = tf.reduce_sum(tf.slice(tf.square(output-targets), (None, 0), [None, 1]))
batch_size = 10
nerve_exists = tf.slice(targets, [0,0], [batch_size, 1])
nerve_exists_prediction = tf.slice(output, [0,0], [batch_size, 1])
loss_func = tf.reduce_sum(nerve_exists*tf.square(output-targets) + (1-nerve_exists)*tf.square(nerve_exists_prediction-nerve_exists))
train = tf.train.AdamOptimizer(0.0001).minimize(loss_func)

session = tf.Session()
session.run(tf.initialize_all_variables())

for i in range(50):
   total_loss = 0
   for j in range(0, len(images), batch_size):
       _, loss = session.run([train, loss_func], feed_dict={inputs : images[j:j+batch_size],
                                                            targets : image_targets[j:j+batch_size],
                                                            #batch_size_placeholder: batch_size]
                                                            })
       total_loss += loss
   print total_loss
