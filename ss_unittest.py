def test_image(img_path): 
    import os
    import tensorflow as tf
    import matplotlib.pyplot as plt

    img = tf.io.read_file(img_path)
    _, ext = os.path.splitext(img_path)
    if ext == ".png":
        img = tf.image.decode_png(img)
        img = tf.image.convert_image_dtype(img, tf.uint8)
    elif ext == ".jpg":
        img = tf.image.decode_jpeg(img)
        img = tf.image.convert_image_dtype(img, tf.uint8)
    else:
        raise ValueError
        
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    
    
def test_ss_dataset(enhanced=False):
    import tensorflow as tf
    import ss_dataset
    import matplotlib.pyplot as plt
    ds = ss_dataset.VOCDataset(r"C:\Users\ming\tinyss\dataset_tfrecords")

    image, label = next(iter(ds(batch_size=1)))
    print(image.shape)
    print(label.shape)
    
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    ax1.imshow(tf.squeeze(image, axis=0))
    ax1.set_title("image")
    ax1.axis("off")   # hide tick axes
    
    if enhanced:
        import cv2
        label = tf.cast(label, tf.uint8)
        label = tf.squeeze(label, axis=0)
        label = cv2.equalizeHist(label.numpy())
        ax2.imshow(label)
    else:
        ax2.imshow(tf.squeeze(label, axis=0))
    ax2.set_title("label")
    ax2.axis("off")

    plt.tight_layout() # auto adjust spacing
    plt.show()


def test_model():
    import os
    import tensorflow as tf
    import ss_model
    
    x = tf.random.normal((2, 240, 320, 3), dtype=tf.float32)
    # model = tf.function(ss_model.Model(16))
    model = ss_model.Model(16)
        
    log_dir = r"logdir\model"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    summary_writer = tf.summary.create_file_writer(log_dir)

    with summary_writer.as_default():
        tf.summary.trace_on(graph=True, profiler=False)
        y = model(x)
        tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)
        tf.summary.trace_off()
        
    print(y.shape)


def test_model2():
    from pathlib import Path
    import tensorflow as tf
    import ss_model

    log_dir = Path("logdir/model")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate model and wrap with explicit input signature
    input_spec = tf.TensorSpec(shape=[None, 240, 320, 3], dtype=tf.float32)
    model = tf.function(ss_model.Model(16), input_signature=[input_spec])

    # 2. Execute once to build the graph
    x = tf.random.normal((2, 240, 320, 3), dtype=tf.float32)
    model(x)

    # 3. Write graph directly from the concrete function
    summary_writer = tf.summary.create_file_writer(str(log_dir))
    with summary_writer.as_default():
        tf.summary.graph(model.get_concrete_function().graph)


# test_image(r"C:\Users\ming\Downloads\VOCdevkit\VOC2012\SegmentationClass\2007_000032.png")
# test_ss_dataset(True)
test_model2()
