import tensorflow as tf
import random
import numpy as np
from datetime import datetime

'''
configure
'''
batch_size = 128
learning_rate = 0.001
embedding_dim = 2
data_path = './norm_test_data.txt'

# no need to define,will be assigned by prepare_data function
field_num = 0
feature_num = 0

tf.logging.set_verbosity(tf.logging.INFO)

def prepare_data(file_path=data_path):
    """
    :param file_path:
    :return: a tuple (data_set,feature2field)
    data_set is a list,each element is a list,the last is label
    """
    feature2field = {}
    data_set = []
    global field_num
    global feature_num
    for sample in open(file_path, 'r'):
        sample_data = []
        field_features = sample.split()[1:]
        for field_feature_pair in field_features:
            feature = int(field_feature_pair.split(':')[1])
            field = int(field_feature_pair.split(':')[0])
            value = float(field_feature_pair.split(':')[2])
            if (field + 1 > field_num):
                field_num = field + 1
            if (feature + 1 > feature_num):
                feature_num = feature + 1
            feature2field[feature] = field
            sample_data.append('{}:{}'.format(feature, value))
        sample_data.append(int(sample[0]))
        data_set.append(sample_data)
    return data_set, feature2field


class FFM:
    def __init__(self, batch_size, learning_rate, embedding_dim,
                 data_path, field_num,
                 feature_num, feature2field, data_set):
        self.batch_size = batch_size
        self.lr = learning_rate
        self.embedding_dim = embedding_dim
        self.data_path = data_path
        self.field_num = field_num
        self.feature_num = feature_num
        self.feature2field = feature2field
        self.data_set = data_set

        with tf.name_scope('embedding_matrix'):
            # a tensor of shape [feature_num] to hold each Wi
            self.linear_weight = tf.get_variable(name='linear_weight',
                                                shape=[self.feature_num],
                                                dtype=tf.float32,
                                                initializer=tf.truncated_normal_initializer(stddev=0.01))
            tf.summary.histogram('linear_weight', self.linear_weight)

            self.quad_weight = tf.get_variable(name='quad_weight',
                                                shape=[self.feature_num, self.field_num, self.embedding_dim],
                                                dtype=tf.float32,
                                                initializer=tf.truncated_normal_initializer(stddev=0.01))
            tf.summary.histogram('quad_weight', self.quad_weight)

        with tf.name_scope('input'):
            self.label = tf.placeholder(tf.float32, shape=[self.batch_size])
            self.feature_value = tf.placeholder(tf.float32, shape=[self.batch_size, self.feature_num])
            
        with tf.name_scope('network'):
            # predict = b0 + sum(Vi * feature_i) + sum(Vij * Vji * feature_i * feature_j)

            # b0:constant bias
            self.b0 = tf.get_variable(name='bias_0', shape=[1], dtype=tf.float32)
            tf.summary.histogram('b0', self.b0)

            # calculate linear term
            self.linear_term = tf.reduce_sum(tf.multiply(self.feature_value, self.linear_weight), axis=1)

            # calculate quadratic term
            self.quad_term = tf.get_variable(name='quad_term', shape=[self.batch_size], dtype=tf.float32)
            for f1 in xrange(0, feature_num - 1):
                for f2 in xrange(f1 + 1, feature_num):
                    W1 = self.quad_weight[f1, self.feature2field[f2]]
                    W2 = self.quad_weight[f2, self.feature2field[f1]]
                    tf.assign_add(self.quad_term, tf.scalar_mul(tf.tensordot(W1, W2, 1), tf.multiply(self.feature_value[:, f1], self.feature_value[:, f2])))


            self.predict = self.b0 + self.linear_term + self.quad_term
            self.losses = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.predict))
            tf.summary.scalar('losses', self.losses)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='Adam')
            self.grad = self.optimizer.compute_gradients(self.losses)
            self.opt = self.optimizer.apply_gradients(self.grad)

        self.sess = tf.InteractiveSession()

        with tf.name_scope('plot'):
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter('./train_plot', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.loop_step = 0


    def step(self):
        '''
        :return: log_loss
        '''
        self.loop_step += 1
        feature, label = self.get_data()
        # feed value to placeholder
        feed_dict = {self.label : label,
                    self.feature_value : feature}
        # feed_dict[self.label] = label
        # arr_feature = np.transpose(np.array(feature))
        # for idx in xrange(0, self.feature_num):
        #     feed_dict[self.feature_value[idx]] = arr_feature[idx]
        _,summary, loss_value = self.sess.run([self.opt,self.merged, self.losses], feed_dict=feed_dict)
        #self.train_writer.add_summary(summary, self.step)
        self.writer.add_summary(summary, self.loop_step)
        return loss_value

    def get_data(self):
        """
        :return: a tuple of feature and label
        feature: shape[batch_size ,feature_num] each element is a sclar
        label:[batch_size] each element is 0 or 1
        """
        feature = []
        label = []
        for _ in xrange(0, self.batch_size):
            t_feature = [0.0] * feature_num
            sample = self.data_set[random.randint(0, len(self.data_set) - 1)]
            label.append(sample[-1])
            sample = sample[:-1]
            for f in sample:
                t_feature[int(f.split(':')[0])] = float(f.split(':')[1])
            feature.append(t_feature)
        return feature, label


if __name__ == "__main__":
    data_set, feature_map = prepare_data(file_path=data_path)
    print("feature num {} field num {}".format(feature_num, field_num))
    tf.logging.info("start building model ({})".format(datetime.now()))
    ffm = FFM(batch_size, learning_rate, embedding_dim, data_path, field_num, feature_num, feature_map, data_set)
    tf.logging.info("model built successfully! ({})".format(datetime.now()))
    for loop in xrange(0, 100000):
        losses = ffm.step()
        if loop % 50 == 0:
            tf.logging.info("loop:{} losses:{} ({})".format(loop, losses, datetime.now()))
