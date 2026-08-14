import numpy as np
import tensorflow as tf
import sonnet as snt
from functools import partial

"""
 |      |                                          |       |       |         |
 |      |                  ========>               |       |       |         |
 |      | ----                                -----|       |       |         |
C=3   C=f[0] |                                |  C=f[0]  C=f[1]  C=f[2] C=#Classes+1
             |                                | 
             |                                |
         |      |                          |       |
         |      |          ========>       |       |
         |      |-------              -----|       |
       C=f[1] C=f[1]   |              |  C=f[1]  C=f[1]
                       |------   -----|
                             |   |
                             |   |
                          |         |
                          |   ==>   |   
                          |         |           
                        C=f[2]    C=f[2]                 
"""

class Model(snt.Module):
    def __init__(self, num_classes, filters=[16, 32, 64], name="model"):
        super(Model, self).__init__(name=name)
        
        self._l1_pre_conv1 = snt.Conv2D(filters[0], 3, name="l1_pre_conv1")
        self._l1_pre_conv1_relu = partial(tf.nn.relu, name="l1_pre_conv1_relu")
        
        self._l1_post_conv1 = snt.Conv2D(filters[0], 3, name="l1_post_conv1")
        self._l1_post_conv1_relu = partial(tf.nn.relu, name="l1_post_conv1_relu")
        
        self._l1_post_conv2 = snt.Conv2D(filters[1], 3, name="l1_post_conv2")
        self._l1_post_conv2_relu = partial(tf.nn.relu, name="l1_post_conv2_relu")

        self._l1_post_conv3 = snt.Conv2D(filters[2], 3, name="l1_post_conv3")
        self._l1_post_conv3_relu = partial(tf.nn.relu, name="l1_post_conv3_relu")

        self._l1_post_conv4 = snt.Conv2D(num_classes, 3, name="l1_post_conv4")


        self._max_pool_l1_l2 = partial(tf.nn.max_pool2d, ksize=2, strides=2, padding="SAME", name="max_pool_l1_l2")
        
        self._l2_pre_conv1 = snt.Conv2D(filters[1], 3, name="l2_pre_conv1")
        self._l2_pre_conv1_relu = partial(tf.nn.relu, name="l2_pre_conv1_relu")
        self._l2_pre_conv2 = snt.Conv2D(filters[1], 3, name="l2_pre_conv2")
        self._l2_pre_conv2_relu = partial(tf.nn.relu, name="l2_pre_conv2_relu")
        self._l2_post_conv1 = snt.Conv2D(filters[1], 3, name="l2_post_conv1")
        self._l2_post_conv1_relu = partial(tf.nn.relu, name="l2_post_conv1_relu")
        self._l2_post_conv2 = snt.Conv2DTranspose(filters[1], 3, stride=2, name="l2_post_conv2")
        self._l2_post_conv2_relu = partial(tf.nn.relu, name="l2_post_conv2")

        self._max_pool_l2_l3 = partial(tf.nn.max_pool2d, ksize=2, strides=2, padding="SAME", name="max_pool_l2_l3")
        
        self._l3_pre_conv1 = snt.Conv2D(filters[2], 3, name="l3_pre_conv1")
        self._l3_pre_conv1_relu = partial(tf.nn.relu, name="l3_pre_conv1_relu")
        self._l3_post_conv1 = snt.Conv2DTranspose(filters[2], 3, stride=2, name="l3_post_conv1")
        self._l3_post_conv1_relu = partial(tf.nn.relu, name="l3_post_conv1_relu")
        
        
    def __call__(self, x):
        x_l1_pre_1 = self._l1_pre_conv1(x)
        x_l1_pre_1 = self._l1_pre_conv1_relu(x_l1_pre_1)        

        x_l2_pre_0 = self._max_pool_l1_l2(x_l1_pre_1)        
        x_l2_pre_1 = self._l2_pre_conv1(x_l2_pre_0)
        x_l2_pre_1 = self._l2_pre_conv1_relu(x_l2_pre_1)
        x_l2_pre_2 = self._l2_pre_conv2(x_l2_pre_1)
        x_l2_pre_2 = self._l2_pre_conv2_relu(x_l2_pre_2)
        x_l3_pre_0 = self._max_pool_l2_l3(x_l2_pre_2)
        
        x_l3_pre_1 = self._l3_pre_conv1(x_l3_pre_0)
        x_l3_pre_1 = self._l3_pre_conv1_relu(x_l3_pre_1)
        x_l3_post_1 = self._l3_post_conv1(x_l3_pre_1)
        x_l3_post_1 = self._l3_post_conv1_relu(x_l3_post_1)
        
        x_l2_post_0 = tf.concat([x_l2_pre_2, x_l3_post_1], axis=-1)
        x_l2_post_1 = self._l2_post_conv1(x_l2_post_0)
        x_l2_post_1 = self._l2_post_conv1_relu(x_l2_post_1)
        x_l2_post_2 = self._l2_post_conv2(x_l2_post_1)
        x_l2_post_2 = self._l2_post_conv2_relu(x_l2_post_2)

        x_l1_post_0 = tf.concat([x_l1_pre_1, x_l2_post_2], axis=-1)
        x_l1_post_1 = self._l1_post_conv1(x_l1_post_0)
        x_l1_post_1 = self._l1_post_conv1_relu(x_l1_post_1)
        
        x_l1_post_2 = self._l1_post_conv2(x_l1_post_1)
        x_l1_post_2 = self._l1_post_conv2_relu(x_l1_post_2)

        x_l1_post_3 = self._l1_post_conv3(x_l1_post_2)
        x_l1_post_3 = self._l1_post_conv3_relu(x_l1_post_3)

        x_l1_post_4 = self._l1_post_conv4(x_l1_post_3)
        
        return x_l1_post_4

