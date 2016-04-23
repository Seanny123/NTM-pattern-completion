import tensorflow as tf

from tasks import *
from utils import pp

flags = tf.app.flags
flags.DEFINE_string("task", "copy", "Task to run [copy, recall, predict]")
flags.DEFINE_integer("epoch", 100000, "Epoch to train [100000]")
flags.DEFINE_integer("input_dim", 10, "Dimension of input [10]")
flags.DEFINE_integer("output_dim", 10, "Dimension of output [10]")
flags.DEFINE_integer("min_length", 1, "Minimum length of input sequence [1]")
flags.DEFINE_integer("max_length", 5, "Maximum length of output sequence [5]")
flags.DEFINE_integer("controller_layer_size", 1, "The size of LSTM controller [1]")
flags.DEFINE_integer("write_head_size", 1, "The number of write head [1]")
flags.DEFINE_integer("read_head_size", 1, "The number of read head [1]")
flags.DEFINE_integer("test_max_length", 120, "Maximum length of output sequence [120]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    with tf.device('/gpu:0'), tf.Session() as sess:
        if FLAGS.task == 'copy':
            if FLAGS.is_train:
                cell, ntm = copy_train(FLAGS, sess)
            else:
                cell = NTMCell(input_dim=FLAGS.input_dim,
                               output_dim=FLAGS.output_dim,
                               controller_layer_size=FLAGS.controller_layer_size,
                               write_head_size=FLAGS.write_head_size,
                               read_head_size=FLAGS.read_head_size)
                ntm = NTM(cell, sess, 1, FLAGS.max_length,
                          test_max_length=FLAGS.test_max_length, forward_only=True)

            ntm.load(FLAGS.checkpoint_dir, 'copy')

            copy(ntm, FLAGS.test_max_length*1/3, sess)
            print
            copy(ntm, FLAGS.test_max_length*2/3, sess)
            print
            copy(ntm, FLAGS.test_max_length*3/3, sess)
        elif FLAGS.task == 'recall':
            if FLAGS.is_train:
                cell, ntm = recall_train(FLAGS, sess)
            else:
                cell = NTMCell(input_dim=FLAGS.input_dim,
                               output_dim=FLAGS.output_dim,
                               controller_layer_size=FLAGS.controller_layer_size,
                               write_head_size=FLAGS.write_head_size,
                               read_head_size=FLAGS.read_head_size)
                ntm = NTM(cell, sess, 1, FLAGS.max_length,
                          test_max_length=FLAGS.test_max_length, forward_only=True)

            ntm.load(FLAGS.checkpoint_dir, 'recall')

            recall(ntm, FLAGS.test_max_length*1/3, sess)
            print
            recall(ntm, FLAGS.test_max_length*2/3, sess)
            print
            recall(ntm, FLAGS.test_max_length*3/3, sess)
        elif FLAGS.task == 'predict':
            if FLAGS.is_train:
                cell, ntm = predict_train(FLAGS, sess)
            else:
                cell = NTMCell(input_dim=FLAGS.input_dim,
                               output_dim=FLAGS.output_dim,
                               controller_layer_size=FLAGS.controller_layer_size,
                               write_head_size=FLAGS.write_head_size,
                               read_head_size=FLAGS.read_head_size)
                ntm = NTM(cell, sess, 1, FLAGS.max_length,
                          test_max_length=FLAGS.test_max_length, forward_only=True)

            ntm.load(FLAGS.checkpoint_dir, 'predict')

            predict(ntm, FLAGS.test_max_length*1/3, sess)
            print
            predict(ntm, FLAGS.test_max_length*2/3, sess)
            print
            predict(ntm, FLAGS.test_max_length*3/3, sess)

if __name__ == '__main__':
    tf.app.run()
