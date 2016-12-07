import numpy as np
import matplotlib.pyplot as plt
import glob, os, os.path
import cv2


# In[2]:

img_dir = "/Users/Will/Documents/DataScience/Kaggle/Ultrasound Nerve Segmentation/"

f_ultrasounds = [img for img in glob.glob(img_dir+"train/*.tif") if 'mask' not in img]
images = [ plt.imread(f_ultrasound) for f_ultrasound in f_ultrasounds[0:50] ]


# In[3]:

f_masks = [img for img in glob.glob(img_dir+"train/*.tif") if 'mask' in img]
masks = [ cv2.imread(mfile, -1) for mfile in f_masks[0:50] ]


# In[4]:

def get_ellipse(mask):
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    im, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    has_ellipse = len(contours) > 0
    if has_ellipse:
        cnt = contours[0]
        ellipse = cv2.fitEllipse(cnt)
        return [ float(has_ellipse) ] + list( ellipse[0] )
    else:
        return [ 0, 0, 0 ]


# In[5]:

image_targets = [ get_ellipse(mask) for mask in masks ]


# In[9]:

images[0]


# In[6]:

import tensorflow as tf
targets = tf.placeholder("float", (None, 3))
inputs = tf.placeholder("float", (None, 420, 580))

# network here

feed_forward_weights_1 = tf.Variable(tf.truncated_normal([420 * 580, 256], stddev=0.01))
feed_forward_bias_1 = tf.Variable(tf.constant(0.0, shape=[256]))

feed_forward_weights_2 = tf.Variable(tf.truncated_normal([256, 3], stddev=0.01))
feed_forward_bias_2 = tf.Variable(tf.constant(0.0, shape=[3]))

input_flat = tf.reshape(inputs, [-1, 420 * 580])

hidden_activations = tf.nn.relu(
           tf.matmul(input_flat, feed_forward_weights_1) + feed_forward_bias_1)


output = tf.nn.relu(
           tf.matmul(hidden_activations, feed_forward_weights_2) + feed_forward_bias_2)


loss_func = tf.reduce_mean(tf.square(output-targets))
train = tf.train.AdamOptimizer(0.1).minimize(loss_func)

session = tf.Session()
session.run(tf.initialize_all_variables())

batch_size = 2
for i in range(1):
    for j in range(0, len(images), batch_size):
        _train, loss = session.run([train, loss_func], feed_dict={inputs : images[j:j+batch_size], targets : image_targets[j:j+batch_size]})
        print(loss)


# In[8]:

get_ipython().magic('pinfo range')


# In[11]:

image_targets[1]


# In[ ]:
