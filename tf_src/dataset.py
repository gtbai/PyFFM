import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

class Dataset(object):

	def __init__(self, data_path, start_ratio, end_ratio):
		super(Dataset, self).__init__()
		self.data_set, self.feature2field, self.field_num, self.feature_num = Dataset.parse_data(data_path, start_ratio, end_ratio)
		self.X_field, self.X_feature, self.X_val, self.Y = self.data_set
		self.batch_size = FLAGS.batch_size
		self.sample_num = len(self.X_field)
		self.idx_perm = np.random.permutation(self.sample_num)
		self.batch_start = 0

	def shuffle_data(self):
		self.idx_perm = np.random.permutation(self.idx_perm)
		self.batch_start = 0

	def next_batch(self):
		batch_end = min(self.batch_start + self.batch_size, self.sample_num)
		idxs = self.idx_perm[self.batch_start : batch_end]
		self.batch_start = batch_end
		return self.X_field[idxs, :], self.X_feature[idxs, :], self.X_val[idxs, :], self.Y[idxs]

	def get_all(self):
		return self.X_field, self.X_feature, self.X_val, self.Y

	@classmethod
	def parse_data(cls, data_path, start_ratio, end_ratio):
		field_num = 0
		feature_num = 0

		count = 0

		feature2field = {}
		X_field, X_feature, X_val, Y = [], [], [], []

		data_file = open(data_path, 'r')
		samples = data_file.readlines()
		samples = samples[int(start_ratio*len(samples)) : int(end_ratio*len(samples))]

		for sample in samples:

			parts = sample.split(' ')
			y = int(parts[0])

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

			# count += 1
			# if count == 1000:
			# 	break

		X_field, X_feature, X_val, Y = np.array(X_field), np.array(X_feature), np.array(X_val), np.array(Y)
		data_set = (X_field, X_feature, X_val, Y)
		return data_set, feature2field, field_num, feature_num