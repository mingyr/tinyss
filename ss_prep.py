import os
import numpy as np
import tensorflow as tf

VOC_DIR = r"C:\Users\ming\Downloads\VOCdevkit\VOC2012"
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 
               'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels."""
    colormap2label = np.zeros(256 ** 3, dtype=np.int32)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[colormap[0] * 256 * 256 + colormap[1] * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices."""
    rgb = tf.cast(colormap, tf.int32)
    idx = colormap[:, :, 0] * 256 * 256 + colormap[:, :, 1] * 256 + colormap[:, :, 2]
    return colormap2label[idx]

def float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))   

def proc(filenames, colormap2label, writer):
    """Read all VOC feature and label images."""
    def map_fn(image, label, height=240, width=320):
        # print(f"image.shape => {image.shape}")
        # print(f"label.shape => {label.shape}")
        
        im_shape = tf.shape(image)
        img_h, img_w = im_shape[0], im_shape[1]
        
        if img_h < height or img_w < width:
            return image, label, False
        else:    
            top = tf.random.uniform([], minval=0, 
                                    maxval=im_shape[0] - height + 1, dtype=tf.int32)
            left = tf.random.uniform([], minval=0, 
                                     maxval=im_shape[1] - width + 1, dtype=tf.int32)
            image = tf.image.crop_to_bounding_box(image, top, left, height, width)
            label = tf.image.crop_to_bounding_box(label, top, left, height, width)
        
        return image, label, True
    
    count = 0    
    for i, fname in enumerate(filenames):
        img = tf.io.read_file(os.path.join(VOC_DIR, 'JPEGImages', f'{fname}.jpg'))
        img = tf.image.decode_jpeg(img)
        img = tf.image.convert_image_dtype(img, tf.float32) 

        lb = tf.io.read_file(os.path.join(VOC_DIR, 'SegmentationClass' ,f'{fname}.png'))
        lb = tf.image.decode_png(lb)
        lb = tf.image.convert_image_dtype(lb, tf.uint8)
        lb = tf.cast(lb, tf.int32)
        lb = tf.expand_dims(voc_label_indices(lb, colormap2label), axis=-1)

        image, label, res = map_fn(img, lb)
        
        if res == False:
            continue
            
        feature = {
            'image': float_feature(np.reshape(image, [-1])),
            'label': float_feature(np.reshape(label, [-1]))
        }

        example = tf.train.Example(features = tf.train.Features(feature = feature))
    
        writer.write(example.SerializeToString())
        count = count + 1
        
    return count

def load_filenames(is_train):
    txt_fname = os.path.join(VOC_DIR, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        files = f.read().split()
    return files

def main():
    save_path = "dataset_tfrecords"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    colormap2label = voc_colormap2label()
    print('beginning prepare VOC tfrecords for training')
    filenames = load_filenames(True)
    writer = tf.io.TFRecordWriter(os.path.join(save_path, 'voc-train.tfr'))
    count = proc(filenames, colormap2label, writer)
    writer.close()
    print('end of tfrecords preparation for training')
    print('#tfrecords for training: {}'.format(count))
    
    print('beginning prepare VOC tfrecords for validation')
    filenames = load_filenames(False)
    writer = tf.io.TFRecordWriter(os.path.join(save_path, 'voc-val.tfr'))
    count = proc(filenames, colormap2label, writer)
    writer.close()
    print('end of tfrecords preparation for validation')
    print('#tfrecords for validation: {}'.format(count))
    
if __name__ == "__main__":
    main()   
