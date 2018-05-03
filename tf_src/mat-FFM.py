import tensorflow as tf
import numpy as np
from datetime import datetime
from math import sqrt

from dataset import Dataset

FLAGS = tf.app.flags.FLAGS

'''
configure
'''
tf.app.flags.DEFINE_integer('embedding_dim', 4, 'embedding dimension (k)')
tf.app.flags.DEFINE_float('regu_param', 2e-5, 'regularization parameter (lambda)')
tf.app.flags.DEFINE_float('learning_rate', 0.2, 'learning rate (eta)')
tf.app.flags.DEFINE_integer('batch_size', 1, 'batch size for mini-batch SGD')

tf.app.flags.DEFINE_string('train_path', '../data/criteo.tr.r100.gbdt0.ffm', 'file path of training dataset')
tf.app.flags.DEFINE_string('test_path', '../data/criteo.va.r100.gbdt0.ffm', 'file path of test dataset')

tf.app.flags.DEFINE_integer('epoch_num', 30, 'number of training epochs')
tf.app.flags.DEFINE_integer('output_inverval_steps', 1, 'number of inverval steps to output training loss')

class FFM:
    def __init__(self, train_set, test_set):
        self.embedding_dim = FLAGS.embedding_dim
        self.regu_param = FLAGS.regu_param
        self.learning_rate = FLAGS.learning_rate

        self.train_set = train_set
        self.test_set = test_set
        self.field_num = max(train_set.field_num, test_set.field_num)
        self.feature_num = max(train_set.feature_num, test_set.feature_num)

        with tf.name_scope('embedding_matrix'):
            # a tensor of shape [feature_num] to hold each Wi
            self.linear_weight = tf.get_variable(name='linear_weight',
                                                shape=[self.feature_num],
                                                dtype=tf.float32,
                                                initializer=tf.random_uniform_initializer(minval=0, maxval=1/sqrt(self.embedding_dim)))
            tf.summary.histogram('linear_weight', self.linear_weight)

            self.quad_weight = tf.get_variable(name='quad_weight',
                                                shape=[self.feature_num, self.field_num, self.embedding_dim],
                                                dtype=tf.float32,
                                                initializer=tf.random_uniform_initializer(minval=0, maxval=1/sqrt(self.embedding_dim)))
            tf.summary.histogram('quad_weight', self.quad_weight)

        with tf.name_scope('input'):
            # self.label = tf.placeholder(tf.float32, shape=[self.batch_size])
            # self.feature_value = tf.placeholder(tf.float32, shape=[self.batch_size, self.feature_num])
            self.X_field = tf.placeholder(tf.int32, shape=[None, self.field_num])
            self.X_feature = tf.placeholder(tf.int32, shape=[None, self.field_num])
            self.X_val = tf.placeholder(tf.float32, shape=[None, self.field_num])
            self.Y = tf.placeholder(tf.float32, shape=[None])
            self.batch_size = tf.shape(self.X_field)[0]

        with tf.name_scope('network'):
            # predict = b0 + sum(Vi * feature_i) + sum(Vij * Vji * feature_i * feature_j)

            # b0:constant bias
            self.b0 = tf.get_variable(name='bias_0', shape=[1], dtype=tf.float32)
            tf.summary.histogram('b0', self.b0)

            # calculate linear term
            self.linear_term = tf.reduce_sum(tf.multiply(self.X_val, tf.gather(self.linear_weight, self.X_feature)), axis=1)

            # calculate quadratic term
            feature_repeated = tf.reshape(tf.tile(tf.expand_dims(self.X_feature, -1), [1, 1, self.field_num]), shape=[self.batch_size, self.field_num*self.field_num])

            field_tiled = tf.tile(self.X_field, [1, self.field_num])
            pair_idx = tf.stack([feature_repeated, field_tiled], axis=-1)
            pair_idx_flattened = tf.reshape(pair_idx, [-1, 2])
            pair_weight = tf.reshape(tf.gather_nd(self.quad_weight, pair_idx_flattened), shape=[self.batch_size, self.field_num, self.field_num, self.embedding_dim])
            X_val_repeated = tf.reshape(tf.tile(tf.expand_dims(self.X_val, -1), [1, 1, self.field_num]), shape=[self.batch_size, self.field_num, self.field_num, 1])
            weight_val = tf.multiply(pair_weight, X_val_repeated)

            # triu = tf.constant(np.expand_dims(np.stack([np.triu(np.ones((self.field_num, self.field_num)), k=1) for _ in range(self.batch_size)], axis=0), -1), dtype=tf.float32)
            triu = tf.expand_dims(tf.tile(tf.constant(np.expand_dims(np.triu(np.ones((self.field_num, self.field_num)), k=1), axis=0), dtype=tf.float32), [self.batch_size, 1, 1]), axis=-1)

            weight_val_transped = tf.transpose(weight_val, [0, 2, 1, 3])

            self.quad_term = tf.reduce_sum(tf.multiply(tf.multiply(weight_val, triu), tf.multiply(weight_val_transped, triu)), axis=[1, 2, 3])

            self.predict = self.b0 + self.linear_term + self.quad_term
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.predict))
            tf.summary.scalar('losses', self.loss)

            self.regu = tf.nn.l2_loss(self.linear_weight) + tf.nn.l2_loss(self.quad_weight)
            self.loss_with_regu = self.loss + self.regu_param * self.regu
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='Adam')
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.train_op = self.optimizer.minimize(self.loss_with_regu, global_step=self.global_step)

        self.sess = tf.InteractiveSession()

        with tf.name_scope('plot'):
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter('./train_plot', self.sess.graph)
            self.test_writer = tf.summary.FileWriter('./test_plot', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.loop_step = 0


    def step(self):
        '''
        :return: log_loss
        '''
        self.loop_step += 1
        X_field_batch, X_feature_batch, X_val_batch, Y_batch = self.get_data()
        # feed value to placeholder
        feed_dict = {self.X_field : X_field_batch,
                    self.X_feature : X_feature_batch,
                    self.X_val : X_val_batch,
                    self.Y : Y_batch}
        _, summary, loss_value = self.sess.run([self.opt, self.merged, self.losses], feed_dict=feed_dict)
        self.writer.add_summary(summary, self.loop_step)
        return loss_value

    def train(self):
        for epoch in range(FLAGS.epoch_num):
            self.train_set.shuffle_data()
            while True:
                # train with mini-batch SGD
                X_field_batch, X_feature_batch, X_val_batch, Y_batch = self.train_set.next_batch()
                if len(X_field_batch) == 0:  # if no data in this batch, means this epoch is finished
                    tf.logging.info("Finished epoch {}".format(epoch+1))
                    break
                feed_dict = {
                    self.X_field : X_field_batch,
                    self.X_feature : X_feature_batch,
                    self.X_val : X_val_batch,
                    self.Y : Y_batch
                }
                _, train_loss, train_summary, global_step = self.sess.run([self.train_op, self.loss, self.merged, self.global_step], feed_dict=feed_dict)
                self.train_writer.add_summary(train_summary, global_step)
                if global_step % FLAGS.output_inverval_steps == 0:
                    tf.logging.info("global step:{} training loss:{} ({})".format(global_step, train_loss, datetime.now()))

            # evaluate loss on test set
            X_field_test, X_feature_test, X_val_test, Y_test = self.test_set.get_all()
            feed_dict = {
                self.X_field : X_field_test,
                self.X_feature : X_feature_test,
                self.X_val : X_val_test,
                self.Y : Y_test
            }
            test_loss, test_summary = self.sess.run([self.loss, self.merged], feed_dict = feed_dict)
            self.test_writer.add_summary(test_summary, epoch)
            tf.logging.info("epoch:{} test loss:{} ({})".format(epoch, test_loss, datetime.now()))


def main(unused_args):
    train_set = Dataset(FLAGS.train_path)
    test_set = Dataset(FLAGS.test_path)
    print("feature num {} field num {}".format(train_set.feature_num, train_set.field_num))
    tf.logging.info("start building model ({})".format(datetime.now()))
    ffm = FFM(train_set, test_set)
    tf.logging.info("model built successfully! ({})".format(datetime.now()))
    # for loop in xrange(0, 100000):
    #     losses = ffm.step()
    #     if loop % 1 == 0:
    #         tf.logging.info("loop:{} losses:{} ({})".format(loop, losses, datetime.now()))
    ffm.train()

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
