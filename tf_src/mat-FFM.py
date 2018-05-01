import tensorflow as tf
import numpy as np
from datetime import datetime

'''
configure
'''
batch_size = 128
learning_rate = 0.001
embedding_dim = 2
train_path = '../data/criteo.tr.r100.gbdt0.ffm'

# no need to define,will be assigned by prepare_data function
field_num = 0
feature_num = 0

tf.logging.set_verbosity(tf.logging.INFO)

def prepare_data(file_path=train_path):
    """
    :param file_path: file path of training dataset
    :return: a tuple: data_set, feature2field, where data_set is a tuple of the following form:
        (X_field, X_feature, X_val, Y)
    """
    global field_num
    global feature_num

    feature2field = {}
    X_field, X_feature, X_val, Y = [], [], [], []

    # for sample in open(file_path, 'r'):
    #     sample_data = []
    #     field_features = sample.split()[1:]
    #     for field_feature_pair in field_features:
    #         feature = int(field_feature_pair.split(':')[1])
    #         field = int(field_feature_pair.split(':')[0])
    #         value = float(field_feature_pair.split(':')[2])
    #         if (field + 1 > field_num):
    #             field_num = field + 1
    #         if (feature + 1 > feature_num):
    #             feature_num = feature + 1
    #         feature2field[feature] = field
    #         sample_data.append('{}:{}'.format(feature, value))
    #     sample_data.append(int(sample[0]))
    #     data_set.append(sample_data)

    for sample in open(train_path, 'r'):

        parts = sample.split(' ')
        y = int(parts[0])
        y = y if y == 1 else -1
        x_field = []
        x_feature = []
        x_val = []
        for pair in parts[1:]:

            field = int(pair.split(':')[0])
            feature = int(pair.split(':')[1])
            val = int(pair.split(':')[2])

            x_field.append(field)
            x_feature.append(feature)
            x_val.append(val)

            feature2field[feature] = field

            field_num = max(field_num, field + 1)
            feature_num = max(feature_num, feature + 1)

        X_field.append(x_field)
        X_feature.append(x_feature)
        X_val.append(x_val)
        Y.append(y)

    X_field, X_feature, X_val, Y = np.array(X_field), np.array(X_feature), np.array(X_val), np.array(Y)
    data_set = (X_field, X_feature, X_val, Y)
    return data_set, feature2field


class FFM:
    def __init__(self, batch_size, learning_rate, embedding_dim,
                 data_path, field_num,
                 feature_num, feature2field, train_set):
        self.batch_size = batch_size
        self.lr = learning_rate
        self.embedding_dim = embedding_dim
        self.data_path = data_path
        self.field_num = field_num
        self.feature_num = feature_num
        self.feature2field = feature2field
        self.train_set = train_set
        self.batch_start = 0
        self.idx_perm = np.random.permutation(len(self.train_set[0]))

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
            # self.label = tf.placeholder(tf.float32, shape=[self.batch_size])
            # self.feature_value = tf.placeholder(tf.float32, shape=[self.batch_size, self.feature_num])
            self.X_field = tf.placeholder(tf.float32, shape=[self.batch_size, self.field_num])
            self.X_feature = tf.placeholder(tf.float32, shape=[self.batch_size, self.field_num])
            self.X_val = tf.placeholder(tf.float32, shape=[self.batch_size, self.field_num])
            self.Y = tf.placeholder(tf.float32, shape=[self.batch_size])
            
        with tf.name_scope('network'):
            # predict = b0 + sum(Vi * feature_i) + sum(Vij * Vji * feature_i * feature_j)

            # b0:constant bias
            self.b0 = tf.get_variable(name='bias_0', shape=[1], dtype=tf.float32)
            tf.summary.histogram('b0', self.b0)

            # calculate linear term
            self.linear_term = tf.reduce_sum(tf.multiply(self.X_val, tf.gather(self.linear_weight, self.X_feature)), axis=1)

            # calculate quadratic term
            # self.quad_term = tf.get_variable(name='quad_term', shape=[self.batch_size], dtype=tf.float32)
            # for f1 in xrange(0, feature_num - 1):
            #     for f2 in xrange(f1 + 1, feature_num):
            #         W1 = self.quad_weight[f1, self.feature2field[f2]]
            #         W2 = self.quad_weight[f2, self.feature2field[f1]]
            #         tf.assign_add(self.quad_term, tf.scalar_mul(tf.tensordot(W1, W2, 1), tf.multiply(self.feature_value[:, f1], self.feature_value[:, f2])))
            # repeat X_feature for self.field_num times
            feature_repeated = tf.reshape(tf.tile(tf.expand_dims(self.X_feature, -1), [1, 1, self.field_num]), shape=[self.batch_size, self.field_num*self.field_num])
            field_tiled = tf.tile(self.X_field, [1, self.feature_num])
            pair_idx = tf.stack([feature_repeated, field_tiled], axis=2)
            pair_idx_flattened = tf.reshape(pair_idx, [-1, 2])
            pair_weight = tf.reshape(tf.gather_nd(self.quad_weight, pair_idx_flattened), shape=[self.batch_size, self.field_num, self.field_num, self.embedding_dim])
            X_val_repeated = tf.reshape(tf.tile(tf.expand_dims(self.X_val, -1), [1, 1, self.field_num]), self.batch_size, self.field_num, self.field_num, 1)
            weight_val = tf.multiply(pair_weight, X_val_repeated)

            triu = tf.constant(np.expand_dims(np.stack([np.ones([self.field_num, self.field_num])[np.triu_indices(self.field_num, k=1)] for _ in range(self.batch_size)], axis=0), -1))

            weight_val_transped = tf.transpose(weight_val, [0, 2, 1, 3])
            self.quad_term = tf.reduce_sum(tf.multiply(tf.multiply(weight_val, triu), tf.multiply(weight_val_transped, triu)), axis=[1, 2, 3])

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
        X_field_batch, X_feature_batch, X_val_batch, Y_batch = self.get_data()
        # feed value to placeholder
        feed_dict = {self.X_field : X_field_batch,
                    self.X_feature : X_feature_batch,
                    self.X_val : X_val_batch,
                    self.Y : Y_batch}
        _,summary, loss_value = self.sess.run([self.opt,self.merged, self.losses], feed_dict=feed_dict)
        self.writer.add_summary(summary, self.loop_step)
        return loss_value

    def get_data(self):
        """
        :return: a tuple of feature and label
        feature: shape[batch_size ,feature_num] each element is a sclar
        label:[batch_size] each element is 0 or 1
        """
        # feature = []
        # label = []
        # batch_end = self.batch_start+self.batch_size
        # if batch_end > len(self.data_set):
        #     self.idx_perm = np.random.permutation(self.idx_perm)
        #     self.batch_start = 0
        #     batch_end = self.batch_start+self.batch_size
        # for idx in self.idx_perm[self.batch_start:batch_end]:
        #     t_feature = [0.0] * feature_num
        #     sample = self.data_set[idx]
        #     label.append(sample[-1])
        #     sample = sample[:-1]
        #     for f in sample:
        #         t_feature[int(f.split(':')[0])] = float(f.split(':')[1])
        #     feature.append(t_feature)
        # self.batch_start = batch_end
        # return feature, label

        batch_end = self.batch_start + self.batch_size
        if batch_end > len(self.data_set[0]):
            self.idx_perm = np.random.permutation(self.idx_perm)
            self.batch_start = 0
            batch_end = self.batch_start+self.batch_size
        idxs = self.idx_perm[self.batch_start : batch_end]
        return self.data_set[0][idxs, :], self.data_set[1][idxs, :], self.data_set[2][idxs, :], self.data_set[3][idxs]

if __name__ == "__main__":
    train_set, feature2field = prepare_data(file_path=train_path)
    print("feature num {} field num {}".format(feature_num, field_num))
    tf.logging.info("start building model ({})".format(datetime.now()))
    ffm = FFM(batch_size, learning_rate, embedding_dim, train_path, field_num, feature_num, feature2field, train_set)
    tf.logging.info("model built successfully! ({})".format(datetime.now()))
    for loop in xrange(0, 100000):
        losses = ffm.step()
        if loop % 50 == 0:
            tf.logging.info("loop:{} losses:{} ({})".format(loop, losses, datetime.now()))
