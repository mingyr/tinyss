import os
import numpy as np
import tensorflow as tf
import sonnet as snt

class VOCDataset(snt.Module):
    def __init__(self, tfrec_dir, img_dims=[240, 320], name="voc_dataset"):
        super(VOCDataset, self).__init__(name=name)
        self._tfrec_dir = tfrec_dir
        self._img_dims = img_dims
        
    def __call__(self, batch_size=2, repeats=-1, is_train=True):
        """Read all VOC feature and label images."""
        dims = np.prod(self._img_dims)
        def map_fn(example_proto):  

            feature_description = {
                'image': tf.io.FixedLenFeature([dims*3], tf.float32),
                'label': tf.io.FixedLenFeature([dims*1], tf.float32)}

            example_parsed = tf.io.parse_single_example(example_proto, feature_description)
            image, label = example_parsed['image'], tf.cast(example_parsed['label'], tf.int32)
                 
            return tf.reshape(image, self._img_dims+[3]), tf.reshape(label, self._img_dims+[1])
            
        ds = tf.data.TFRecordDataset(os.path.join(self._tfrec_dir, 
                                                  "voc-train.tfr" if is_train else "voc-val.tfr"))
        ds = ds.map(map_fn)
        ds = ds.repeat(repeats)
        ds = ds.shuffle(buffer_size=100)
        ds = ds.batch(batch_size)
        
        return ds
        