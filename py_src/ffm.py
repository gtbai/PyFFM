import numpy as np


class FFM:
    def __init__(self, latent_dim, reg_parm, batch_size, learning_rate, n_iter):
        self.latent_dim = latent_dim
        self.reg_parm = reg_parm
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y, feature2field: dict):
        n_fields = max(feature2field.values()) + 1
        n_features = max(feature2field.keys()) + 1

        X_val, X_feat = X  # type: (np.ndarray, np.ndarray)
        n_samples, _ = X_val.shape

        self.coef_ = np.random.rand(n_features, n_fields, self.latent_dim) / np.sqrt(self.latent_dim)
        self.g_sum_ = np.ones_like(self.coef_)

        for iteration_idx in range(self.n_iter):
            minibatch_indices = np.random.choice(n_samples, self.batch_size, replace=False)

            self._fit_adagrad((X_val[minibatch_indices], X_feat[minibatch_indices]),
                              y[minibatch_indices], feature2field)

            if iteration_idx % 50 == 0:
                print('Iteration: ' + str(iteration_idx))

    def _fit_adagrad(self, X, y, feature2field: dict):
        X_val, X_feat = X  # type: (np.ndarray, np.ndarray)
        X_field = np.vectorize(feature2field.__getitem__)(X_feat)

        _, n_fields = X_val.shape

        coef_idx = (np.repeat(X_feat, n_fields, axis=-1), np.tile(X_field, n_fields))
        coef = self.coef_[coef_idx[0], coef_idx[1]]

        X_val_repeated = np.repeat(X_val, n_fields, axis=-1)
        coef_val = (coef * X_val_repeated[..., None]).reshape(self.batch_size, n_fields, n_fields, -1)

        triu = np.triu_indices(n_fields, k=1)
        kernel_ffm = np.sum(coef_val.transpose((0, 2, 1, 3))[:, triu[0], triu[1], :]
                            * coef_val[:, triu[0], triu[1], :], axis=(1, 2))

        kappa = (- y / (1 + np.exp(y * kernel_ffm))).reshape(-1, 1, 1, 1)

        coef = coef.reshape(self.batch_size, n_fields, n_fields, -1)
        X_val_repeated = X_val_repeated.reshape(self.batch_size, n_fields, n_fields, 1)

        gradient = self.reg_parm * coef + kappa * coef.transpose((0, 2, 1, 3)) * X_val_repeated * X_val_repeated.transpose((0, 2, 1, 3))
        gradient = gradient.reshape(self.batch_size, n_fields * n_fields, -1)

        np.add.at(self.g_sum_, coef_idx, gradient ** 2)
        np.add.at(self.coef_, coef_idx, - self.learning_rate / np.sqrt(self.g_sum_[coef_idx]) * gradient)

    def predict(self, X, feature2field: dict):
        X_val, X_feat = X
        X_field = np.vectorize(feature2field.__getitem__)(X_feat)

        n_samples, n_fields = X_val.shape

        coef_idx = (np.repeat(X_feat, n_fields, axis=-1), np.tile(X_field, n_fields))
        coef = self.coef_[coef_idx[0], coef_idx[1]]

        X_val_repeated = np.repeat(X_val, n_fields, axis=-1)[..., None]
        coef_val = (coef * X_val_repeated).reshape(n_samples, n_fields, n_fields, -1)

        triu = np.triu_indices(n_fields, k=1)
        kernel_ffm = np.sum(coef_val.transpose((0, 2, 1, 3))[:, triu[0], triu[1], :]
                            * coef_val[:, triu[0], triu[1], :], axis=(1, 2))

        return np.sign(kernel_ffm)