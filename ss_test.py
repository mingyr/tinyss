import os, sys
import numpy as np
import tensorflow as tf
from absl import app
from ss_config import FLAGS
from ss_dataset import VOCDataset


# 训练好的模型的保存路径    
save_path = r"saved_model"

def test(unused_argv):
    # 加载测试数据
    voc_dir = r"C:\Users\ming\tinyss\dataset_tfrecords"
    dataset = Dataset(voc_dir)(batch_size=1, repeats=1, is_train=False)
    
    # 由输入得到模型的输出
    assert os.path.exists(save_path), "模型路径不存在"

    model = tf.saved_model.load(save_path)
    assert len(model.all_variables) > 0, "加载模型失败"

    count = 0
    accu = 0

    for i, (image, label) in enumerate(dataset):
        logit = model.inference(image)

        # 计算损失函数
        logit = tf.math.argmax(logit, axis=-1, keep)

        with tf.control_dependencies([tf.assert_equal(tf.rank(logit), tf.rank(label))]):
            accu += tf.squeeze(tf.cast(tf.math.equal(logit, label), tf.float32))

        count += 1

    print(f"count: {count}, accu: {accu}")
    print("Test accuracy: {:.2f}".format(accu/count))

if __name__ == "__main__":
    app.run(test)

