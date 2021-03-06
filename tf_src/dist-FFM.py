import tensorflow as tf
import numpy as np
from datetime import datetime

from dataset import Dataset

FLAGS = tf.app.flags.FLAGS

'''
configure
'''
tf.app.flags.DEFINE_integer('embedding_dim', 4, 'embedding dimension (k)')
tf.app.flags.DEFINE_float('regu_param', 2e-5, 'regularization parameter (lambda)')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate (eta)')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size for mini-batch SGD')

tf.app.flags.DEFINE_string('train_path', '/mnt/project/kaggle-2014-criteo/tr.ffm', 'file path of training dataset')
tf.app.flags.DEFINE_string('test_path', '/mnt/project/kaggle-2014-criteo/te.ffm', 'file path of test dataset')

tf.app.flags.DEFINE_integer('epoch_num', 15, 'number of training epochs')
tf.app.flags.DEFINE_integer('output_inverval_steps', 1, 'number of inverval steps to output training loss')

tf.app.flags.DEFINE_boolean('early_stop', True, 'whether to early stop during training')
tf.app.flags.DEFINE_float('train_ratio', 0.8, 'ratio of training data in the whole dataset')

tf.app.flags.DEFINE_boolean('save_model', True, 'whether to save model after evaluation')
tf.app.flags.DEFINE_string('ckpt_path', '/mnt/project/ckpt/mat-FFM.ckpt', 'file path of checkpoint (saved model)')

tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')


class FFM:
    def __init__(self, train_set, valid_set, test_set, cluster, is_chief):
        self.embedding_dim = FLAGS.embedding_dim
        self.regu_param = FLAGS.regu_param
        self.learning_rate = FLAGS.learning_rate

        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set

        self.field_num = max(train_set.field_num, test_set.field_num)
        self.feature_num = max(train_set.feature_num, test_set.feature_num)
        if FLAGS.early_stop:
            self.field_num = max(self.field_num, valid_set.field_num)
            self.feature_num = max(self.feature_num, valid_set.feature_num)

        self.cluster = cluster
        self.is_chief = is_chief

        print("field num {} feature num {}".format(self.field_num, self.feature_num))

        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_id, cluster = cluster)):
            with tf.name_scope('embedding_matrix'):
                # a tensor of shape [feature_num] to hold each Wi
                self.linear_weight = tf.get_variable(name='linear_weight',
                                                    shape=[self.feature_num],
                                                    dtype=tf.float32,
                                                    # initializer=tf.random_uniform_initializer(minval=0, maxval=1/sqrt(self.embedding_dim))
                                                    initializer=tf.random_uniform_initializer(minval=0, maxval=0.01)
                            )
                tf.summary.histogram('linear_weight', self.linear_weight)

                self.quad_weight = tf.get_variable(name='quad_weight',
                                                    shape=[self.feature_num, self.field_num, self.embedding_dim],
                                                    dtype=tf.float32,
                                                    # initializer=tf.random_uniform_initializer(minval=0, maxval=1/sqrt(self.embedding_dim))
                                                    initializer=tf.random_uniform_initializer(minval=0, maxval=0.01)
                            )
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
                # tf.summary.histogram('b0', self.b0)

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
                # tf.summary.scalar('losses', self.loss)

                self.regu = tf.nn.l2_loss(self.linear_weight) + tf.nn.l2_loss(self.quad_weight)
                self.loss_with_regu = self.loss + self.regu_param * self.regu
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='Adam')
                # self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1)
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.train_op = self.optimizer.minimize(self.loss_with_regu, global_step=self.global_step)

        # self.sess = tf.train.MonitoredTrainingSession(master=target, is_chief=is_chief, checkpoint_dir=FLAGS.ckpt_path)

        if is_chief:
            self.saver = tf.train.Saver()
        
        # with tf.name_scope('plot'):
        #     self.merged = tf.summary.merge_all()
        #     self.train_writer = tf.summary.FileWriter('/mnt/project/train_plot', self.sess.graph)
        #     self.valid_writer = tf.summary.FileWriter('/mnt/project/valid_plot', self.sess.graph)
        #     self.test_writer = tf.summary.FileWriter('/mnt/project/test_plot', self.sess.graph)


    def train(self, target, is_chief):
        with tf.train.MonitoredTrainingSession(master=target, is_chief=is_chief, checkpoint_dir=FLAGS.ckpt_path) as sess:
            X_field_valid, X_feature_valid, X_val_valid, Y_valid = self.valid_set.get_all()
            feed_dict = {
                        self.X_field : X_field_valid,
                        self.X_feature : X_feature_valid,
                        self.X_val : X_val_valid,
                        self.Y : Y_valid
                    }
            prev_valid_loss = sess.run([self.loss], feed_dict = feed_dict)

            for epoch in range(1, FLAGS.epoch_num+1):
                self.train_set.shuffle_data()
                while True:
                    # train with mini-batch SGD
                    X_field_batch, X_feature_batch, X_val_batch, Y_batch = self.train_set.next_batch()
                    if len(X_field_batch) == 0:  # if no data in this batch, means this epoch is finished
                        tf.logging.info("Finished epoch {}".format(epoch))
                        break
                    feed_dict = {
                        self.X_field : X_field_batch,
                        self.X_feature : X_feature_batch,
                        self.X_val : X_val_batch,
                        self.Y : Y_batch
                    }
                    _, train_loss, global_step = sess.run([self.train_op, self.loss, self.global_step], feed_dict=feed_dict)

                    # self.train_writer.add_summary(train_summary, global_step)
                    if global_step % FLAGS.output_inverval_steps == 0:
                        tf.logging.info("global step:{} training loss:{} ({})".format(global_step, train_loss, datetime.now()))

                if FLAGS.early_stop:
                    # evaluate on validation set, determine whether to early stop
                    X_field_valid, X_feature_valid, X_val_valid, Y_valid = self.valid_set.get_all()
                    feed_dict = {
                        self.X_field : X_field_valid,
                        self.X_feature : X_feature_valid,
                        self.X_val : X_val_valid,
                        self.Y : Y_valid
                    }
                    valid_loss = sess.run([self.loss], feed_dict = feed_dict)
                    # self.valid_writer.add_summary(valid_summary, epoch)
                    tf.logging.info("epoch:{} validation loss:{} ({})".format(epoch, valid_loss, datetime.now()))
                    if valid_loss > prev_valid_loss:
                        tf.logging.info("validation loss goes up after {} epochs".format(epoch))
                        # return epoch - 1 
                    prev_valid_loss = valid_loss

    def test(self):
        # evaluate loss on test set
        X_field_test, X_feature_test, X_val_test, Y_test = self.test_set.get_all()
        feed_dict = {
            self.X_field : X_field_test,
            self.X_feature : X_feature_test,
            self.X_val : X_val_test,
            self.Y : Y_test
        }
        test_loss, test_summary = self.sess.run([self.loss, self.merged], feed_dict = feed_dict)
        tf.logging.info("test loss:{} ({})".format(test_loss, datetime.now()))

    def save_model(self):
        self.saver.save(self.sess, FLAGS.ckpt_path)


def main(unused_args):
    assert FLAGS.job_name in ['ps', 'worker'], 'job_name must be ps or worker'

    # Extract all the hostnames for the ps and worker jobs to construct the
    # cluster spec.
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    tf.logging.info('PS hosts are: %s' % ps_hosts)
    tf.logging.info('Worker hosts are: %s' % worker_hosts)

    cluster = tf.train.ClusterSpec({'ps': ps_hosts,
                                           'worker': worker_hosts})
    server = tf.train.Server(
        {'ps': ps_hosts,
        'worker': worker_hosts},
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_id)

    if FLAGS.job_name == 'ps':
        # `ps` jobs wait for incoming connections from the workers.
        server.join()
    else:
        is_chief = (FLAGS.task_id == 0)
        train_set, valid_set = None, None

        # if is_chief:
        if FLAGS.early_stop:
            train_set = Dataset(FLAGS.train_path, 0.0, FLAGS.train_ratio, True)
            valid_set = Dataset(FLAGS.train_path, FLAGS.train_ratio, 1.0, True)
        else:
            train_set = Dataset(FLAGS.train_path, 0.0, 1.0, True)
        test_set = Dataset(FLAGS.test_path, 0.0, 1.0, False)

        tf.logging.info("start building model ({})".format(datetime.now()))
        ffm = FFM(train_set, valid_set, test_set, cluster, is_chief)
        tf.logging.info("model built successfully! ({})".format(datetime.now()))

        stopping_epoch_num = ffm.train(server.target, is_chief)
        if FLAGS.early_stop:
            if stopping_epoch_num is not None:            
                    ffm.train_set = Dataset(FLAGS.train_path, 0.0, 1.0)
                    FLAGS.early_stop = False
                    FLAGS.epoch_num = stopping_epoch_num if stopping_epoch_num != None else FLAGS.epoch_num
                    ffm.sess.run(tf.global_variables_initializer())
                    ffm.train()
            else:
                tf.logging.info("model with learning rate:{} and batch size:{} does not converge.".format(FLAGS.learning_rate, FLAGS.batch_size))
        ffm.test()
        if is_chief:
            ffm.save_model()

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
