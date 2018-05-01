import numpy as np


feature2field = {}
data_set = []

num_field = 0       # 39
num_feature = 0     # 999997
num_latent_factor = 4
reg = 0.0002
learning_rate = 0.1

counter = 0
for line in open('../data/criteo.tr.r100.gbdt0.ffm'):
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

    data_set.append((features, values, label))

    counter += 1
    if counter == 10000:
        break


num_field = max(feature2field.values()) + 1
num_feature = max(feature2field.keys()) + 1

# np.random.seed(0)
W = np.random.rand(num_feature, num_field, num_latent_factor) * 0.01
acc_gradient = np.ones_like(W)

X_feature = np.array([x[0] for x in data_set])
X = np.array([x[1] for x in data_set])
Y = np.array([x[2] for x in data_set])

N, D = X.shape  # D should be equal to num_field

for e in range(10):

    for i in range(N):
        x = X[i]
        y = Y[i]
        # f = X_feature[0]
        #
        # kernel_ffm = 0
        # for j1 in range(D):
        #     for j2 in range(j1+1, D):
        #         w_j1f2 = W[f[j1], feature2field[f[j2]]]
        #         w_j2f1 = W[f[j2], feature2field[f[j1]]]
        #         kernel_ffm += np.dot(w_j1f2, w_j2f1) * x[j1] * x[j2]

        j = X_feature[i]                        # feature
        f = [feature2field[ji] for ji in j]     # field

        pair_weight = W[np.repeat(j, D), np.tile(f, D)]
        weight_value = (pair_weight * np.repeat(x, D)[:, None]).reshape(D, D, -1)

        tri_indices = np.triu_indices(D, k=1)
        kernel_ffm2 = np.sum(weight_value.transpose((1, 0, 2))[tri_indices] * weight_value[tri_indices])

        kappa = - y / (1 + np.exp(y * kernel_ffm2))

        pair_weight = pair_weight.reshape(D, D, -1)
        gradient = reg * pair_weight + kappa * pair_weight.transpose((1, 0, 2))
        gradient = gradient.reshape((D*D, -1))
        acc_gradient[np.repeat(j, D), np.tile(f, D)] += gradient ** 2

        W[np.repeat(j, D), np.tile(f, D)] -= learning_rate / np.sqrt(acc_gradient[np.repeat(j, D), np.tile(f, D)]) * gradient

    loss = 0
    for k in range(N):
        x = X[k]
        y = Y[k]

        j = X_feature[k]  # feature
        f = [feature2field[ji] for ji in j]  # field

        pair_weight = W[np.repeat(j, D), np.tile(f, D)]
        weight_value = (pair_weight * np.repeat(x, D)[:, None]).reshape(D, D, -1)

        tri_indices = np.triu_indices(D, k=1)
        kernel_ffm2 = np.sum(weight_value.transpose((1, 0, 2))[tri_indices] * weight_value[tri_indices])
        loss += np.log(1 + np.exp(-y * kernel_ffm2))
    print(loss / N)


# def ffm1():
#     f = X_feature[0]
#     kernel_ffm = 0
#     for j1 in range(D):
#         for j2 in range(j1 + 1, D):
#             w_j1f2 = W[f[j1], feature2field[f[j2]]]
#             w_j2f1 = W[f[j2], feature2field[f[j1]]]
#             kernel_ffm += np.dot(w_j1f2, w_j2f1) * x[j1] * x[j2]
#
#
# def ffm2():
#     j = X_feature[0]  # feature
#     f = [feature2field[ji] for ji in j]  # field
#
#     pair_weight = (W[np.repeat(j, D), np.tile(f, D)] * np.repeat(x, D)[:, None]).reshape(D, D, -1)
#
#     tri_indices = np.triu_indices(D, k=1)
#     kernel_ffm2 = np.sum(pair_weight.transpose((1, 0, 2))[tri_indices] * pair_weight[tri_indices])
