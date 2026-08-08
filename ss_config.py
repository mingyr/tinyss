from absl import flags

FLAGS = flags.FLAGS

# Model parameters
flags.DEFINE_integer('num_classes', 21, 'number of classes, channel number for last layer.')

# data specification for data
flags.DEFINE_integer('img_height', 240, 'image height.')
flags.DEFINE_integer('img_width', 320, 'image width.')
flags.DEFINE_integer('num_channels', 3, 'image channels.')


# Training options.
# Optimizer parameters.
flags.DEFINE_float("learning_rate", 0.001, "Optimizer learning rate.")
flags.DEFINE_float("optimizer_epsilon", 1e-10, "Epsilon used for RMSProp optimizer.")

flags.DEFINE_integer("num_epochs", 10, "Number of iterations to train for.")

flags.DEFINE_integer('batch_size', 8, 'batch size for test')
flags.DEFINE_integer('test_size', 100, 'batch size for test')

flags.DEFINE_string('train_output_dir', 'output-train', 'directory for training output.')
flags.DEFINE_string('test_output_dir', 'output-test', 'directory for testing output.')

flags.DEFINE_string('checkpoint_dir', '', 'directory for checkpoint file.')

flags.DEFINE_integer("report_interval", 10, "Iterations between reports (samples, valid loss).")
flags.DEFINE_boolean("sanity_check", False, "sanity check for parameters compatibility")


