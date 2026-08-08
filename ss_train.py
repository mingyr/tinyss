import os
from absl import app
import numpy as np
import tensorflow as tf
import sonnet as snt
from ss_config import FLAGS
from ss_dataset import VOCDataset
from ss_model import Model

    
# 保存模型
def save_model(module):                            
    @tf.function(input_signature=[tf.TensorSpec([None, FLAGS.img_height, FLAGS.img_width, FLAGS.num_channels])])
    def inference(x):
        return module(x)

    save_path = r"saved_model"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    to_save = snt.Module()
    to_save.inference = inference
    to_save.all_variables = list(module.variables)
    tf.saved_model.save(to_save, save_path)

# 训练模型
def main(self):
    voc_dir = r"dataset_tfrecords"
    train_dataset = VOCDataset(voc_dir)(batch_size=FLAGS.batch_size,
                                        repeats=FLAGS.num_epochs, is_train=True)
    val_dataset = VOCDataset(voc_dir)(batch_size=1, repeats=-1, is_train=False)
    val_dataset_iter = iter(val_dataset)
    
    # 建立模型
    model = Model(FLAGS.num_classes)

    # 建立优化器
    optimizer = snt.optimizers.Adam(learning_rate=FLAGS.learning_rate, beta1=0.)

    # 准备存储运行时信息的文件夹与对象
    if not os.path.exists(FLAGS.train_output_dir):
        os.makedirs(FLAGS.train_output_dir)
    summary_writer = tf.summary.create_file_writer(FLAGS.train_output_dir)

    @tf.function
    def loop():
        step = np.int64(0)
        with summary_writer.as_default():
            for images, labels in train_dataset:
                step = step + 1
                # 训练，记录前向传播信息
                with tf.GradientTape() as tape:
                    logits = model(images)
                    labels = tf.one_hot(labels, depth=FLAGS.num_classes)
                    if FLAGS.sanity_check:
                        with tf.control_dependencies([tf.assert_equal(tf.shape(logits), tf.shape(labels))]):
                            loss = tf.math.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
                    else:
                        loss = tf.math.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
                # 计算梯度，并更新权重
                variables = model.trainable_variables
                gradients = tape.gradient(loss, variables)
                optimizer.apply(gradients, variables)
    
                tf.print("iteration:", step, " loss - ", loss)
                tf.summary.scalar('loss', loss, step=step)
                
                # 验证
                if step % FLAGS.report_interval == 0:
                    images_val, labels_val = next(val_dataset_iter)
                    logits_val = model(images_val)
                    logits_val = tf.math.argmax(logits_val, -1, output_type=tf.dtypes.int32)
                    if FLAGS.sanity_check:
                        with tf.control_dependencies([tf.assert_equal(tf.shape(logits_val), tf.shape(labels_val))]):
                            prediction = tf.equal(labels_val, logits_val)
                            accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

                        tf.print("iteration:", step, " accuracy - ", accuracy)          
                        tf.summary.scalar('accuracy', accuracy, step=step)
                break
    loop()
    
    # 保存模型        
    save_model(model)
    
if __name__ == "__main__":
    app.run(main)        


