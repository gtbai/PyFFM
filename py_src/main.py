import numpy as np
from ffm import FFM


feature2field = {}

data_set_train = []
counter = 0
for line in open('/mnt/project/kaggle-2014-criteo/tr.ffm'):
    label = int(line[0])
    if label == 0:
        label = -1
    features = []
    values = []

    for pair in line.split()[1:]:
        field, feature, value = [int(x) for x in pair.split(':')]

        features.append(feature)
        values.append(value)

        feature2field[feature] = field

    data_set_train.append((features, values, label))

#    counter += 1
#    if counter == 10000:
#        break

#data_set_test = []
#counter = 0
#for line in open('C:/Users/chenx/PycharmProjects/ffm/data/criteo.va.r100.gbdt0.ffm'):
#    label = int(line[0])
#    if label == 0:
#        label = -1
#    features = []
#    values = []
#
#    for pair in line.split()[1:]:
#        field, feature, value = [int(x) for x in pair.split(':')]
#
#        features.append(feature)
#        values.append(value)
#
#        feature2field[feature] = field
#
#    data_set_test.append((features, values, label))
#
#    counter += 1
#    if counter == 1000:
#        break

X_feature_train = np.array([x[0] for x in data_set_train])
X_value_train = np.array([x[1] for x in data_set_train])
Y_train = np.array([x[2] for x in data_set_train])

#X_feature_test = np.array([x[0] for x in data_set_test])
#X_value_test = np.array([x[1] for x in data_set_test])
#Y_test = np.array([x[2] for x in data_set_test])

clf = FFM(latent_dim=4, reg_parm=0.00002, batch_size=1024, learning_rate=0.2, n_iter=10)
clf.fit((X_value_train, X_feature_train), Y_train, feature2field)
#Y_pred = clf.predict((X_value_test, X_feature_test), feature2field)
