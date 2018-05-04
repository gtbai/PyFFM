import numpy as np


class FFM:
    def __init__(self, latent_dim, reg_parm, batch_size, learning_rate, n_iter):
        self.latent_dim = latent_dim
        self.reg_parm = reg_parm
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.validation_fraction = 0.15
        self.early_stop = True

    def fit(self, X, y, feature2field: dict):
        n_fields = max(feature2field.values()) + 1
        n_features = max(feature2field.keys()) + 1

        X_val, X_feat = X  # type: (np.ndarray, np.ndarray)
        n_samples, _ = X_val.shape

        # self.intercept = np.random.rand() * 0.01
        # self.linear_coef = np.random.rand(n_features) * 0.01
        self.latent_coef = np.random.rand(n_features, n_fields, self.latent_dim) * 0.01
        # self.intercept_G = 1
        # self.linear_G = np.ones_like(self.linear_coef)
        self.latent_G = np.ones_like(self.latent_coef)
        self.m_ = np.zeros_like(self.latent_coef)
        self.v_ = np.zeros_like(self.latent_coef)

        split_idx = int(n_samples * (1 - self.validation_fraction))
        validation_indices = np.arange(split_idx, n_samples)

        prev_validation_loss = np.Infinity

        for iteration_idx in range(self.n_iter):
            minibatch_indices = np.random.choice(split_idx, self.batch_size, replace=False)

            loss = self._fit_adagrad((X_val[minibatch_indices], X_feat[minibatch_indices]),
                                     y[minibatch_indices], feature2field)
            # loss = self._fit_adam((X_val[minibatch_indices], X_feat[minibatch_indices]),
            #                          y[minibatch_indices], feature2field, iteration_idx + 1)
            print('Iteration: {}, loss = {}'.format(iteration_idx, loss))

            if iteration_idx % (split_idx // self.batch_size) == 0:
                validation_loss = self._loss((X_val[validation_indices], X_feat[validation_indices]),
                                             y[validation_indices], feature2field)
                print('\t\tEpoch: {}, validation loss = {}'.format(iteration_idx // (split_idx // self.batch_size),
                                                                   validation_loss))
                if validation_loss > prev_validation_loss:
                    break
                else:
                    prev_validation_loss = validation_loss



    def _fit_adagrad(self, X, y, feature2field: dict):
        X_val, X_feat = X  # type: (np.ndarray, np.ndarray)
        X_field = np.vectorize(feature2field.__getitem__)(X_feat)

        n_samples, n_fields = X_val.shape

        # linear_term = np.sum(X_val * self.linear_coef[X_feat], axis=1)

        coef_idx = (np.repeat(X_feat, n_fields, axis=-1), np.tile(X_field, n_fields))
        coef = self.latent_coef[coef_idx[0], coef_idx[1]]

        X_val_repeated = np.repeat(X_val, n_fields, axis=-1)
        coef_val = (coef * X_val_repeated[..., None]).reshape(n_samples, n_fields, n_fields, -1)

        triu = np.triu_indices(n_fields, k=1)
        latent_term = np.sum(coef_val.transpose((0, 2, 1, 3))[:, triu[0], triu[1], :]
                             * coef_val[:, triu[0], triu[1], :], axis=(1, 2))

        # model = latent_term + linear_term + self.intercept

        # intercept_kappa = np.mean(- y / (1 + np.exp(y * model)))
        # linear_kappa = - y / (1 + np.exp(y * model))
        latent_kappa = - y / (1 + np.exp(y * latent_term))

        coef = coef.reshape(n_samples, n_fields, n_fields, -1)
        X_val_repeated = X_val_repeated.reshape(n_samples, n_fields, n_fields, 1)

        # linear_grad = linear_kappa.reshape(-1, 1) * X_val

        latent_grad = self.reg_parm * coef + latent_kappa.reshape(-1, 1, 1, 1) * coef.transpose(
            (0, 2, 1, 3)) * X_val_repeated * X_val_repeated.transpose((0, 2, 1, 3))
        latent_grad = latent_grad.reshape(n_samples, n_fields * n_fields, -1)

        # self.intercept_G += intercept_kappa ** 2
        # self.intercept -= self.learning_rate / np.sqrt(self.intercept_G) * intercept_kappa
        #
        # self.linear_G[X_feat] += linear_grad ** 2
        # self.linear_coef[X_feat] -= self.learning_rate / np.sqrt(self.linear_G[X_feat]) * linear_grad

        # np.add.at(self.linear_G, X_feat, linear_grad ** 2)
        # np.add.at(self.linear_coef, X_feat, - self.learning_rate / np.sqrt(self.linear_G[X_feat]) * linear_grad)

        self.latent_G[coef_idx] += latent_grad ** 2
        self.latent_coef[coef_idx] -= self.learning_rate / np.sqrt(self.latent_G[coef_idx]) * latent_grad

        # np.add.at(self.latent_G, coef_idx, latent_grad ** 2)
        # np.add.at(self.latent_coef, coef_idx, - self.learning_rate / np.sqrt(self.latent_G[coef_idx]) * latent_grad)

        loss = np.mean(np.log(1 + np.exp(-y * latent_term)))
        loss += self.reg_parm / 2 * (np.linalg.norm(self.latent_coef) ** 2)
        return loss

    def _fit_adam(self, X, y, feature2field: dict, t):
        X_val, X_feat = X  # type: (np.ndarray, np.ndarray)
        X_field = np.vectorize(feature2field.__getitem__)(X_feat)

        n_samples, n_fields = X_val.shape

        coef_idx = (np.repeat(X_feat, n_fields, axis=-1), np.tile(X_field, n_fields))
        coef = self.latent_coef[coef_idx[0], coef_idx[1]]

        X_val_repeated = np.repeat(X_val, n_fields, axis=-1)
        coef_val = (coef * X_val_repeated[..., None]).reshape(n_samples, n_fields, n_fields, -1)

        triu = np.triu_indices(n_fields, k=1)
        kernel_ffm = np.sum(coef_val.transpose((0, 2, 1, 3))[:, triu[0], triu[1], :]
                            * coef_val[:, triu[0], triu[1], :], axis=(1, 2))

        kappa = (- y / (1 + np.exp(y * kernel_ffm))).reshape(-1, 1, 1, 1)

        coef = coef.reshape(n_samples, n_fields, n_fields, -1)
        X_val_repeated = X_val_repeated.reshape(n_samples, n_fields, n_fields, 1)

        gradient = self.reg_parm * coef + kappa * coef.transpose(
            (0, 2, 1, 3)) * X_val_repeated * X_val_repeated.transpose((0, 2, 1, 3))
        gradient = gradient.reshape(n_samples, n_fields * n_fields, -1)

        beta_1 = 0.9
        beta_2 = 0.999
        self.m_[coef_idx] = beta_1 * self.m_[coef_idx] + (1 - beta_1) * gradient
        self.v_[coef_idx] = beta_2 * self.v_[coef_idx] + (1 - beta_2) * gradient * gradient

        m_t = self.m_[coef_idx] / (1 - beta_1 ** t)
        v_t = self.v_[coef_idx] / (1 - beta_2 ** t)

        self.latent_coef[coef_idx] -= self.learning_rate / (np.sqrt(v_t) + 1e-8) * m_t

        loss = np.mean(np.log(1 + np.exp(-y * kernel_ffm)))
        loss += self.reg_parm / 2 * (np.linalg.norm(self.latent_coef) ** 2)
        return loss

    def _loss(self, X, y, feature2field: dict):
        X_val, X_feat = X
        X_field = np.vectorize(feature2field.__getitem__)(X_feat)

        n_samples, n_fields = X_val.shape

        # linear_term = np.sum(X_val * self.linear_coef[X_feat], axis=1)

        coef_idx = (np.repeat(X_feat, n_fields, axis=-1), np.tile(X_field, n_fields))
        coef = self.latent_coef[coef_idx[0], coef_idx[1]]

        X_val_repeated = np.repeat(X_val, n_fields, axis=-1)[..., None]
        coef_val = (coef * X_val_repeated).reshape(n_samples, n_fields, n_fields, -1)

        triu = np.triu_indices(n_fields, k=1)
        latent_term = np.sum(coef_val.transpose((0, 2, 1, 3))[:, triu[0], triu[1], :]
                            * coef_val[:, triu[0], triu[1], :], axis=(1, 2))

        model = latent_term

        loss = np.mean(np.log(1 + np.exp(-y * model)))
        loss += self.reg_parm / 2 * (np.linalg.norm(self.latent_coef) ** 2)
        return loss

    def predict(self, X, feature2field: dict):
        X_val, X_feat = X
        X_field = np.vectorize(feature2field.__getitem__)(X_feat)

        n_samples, n_fields = X_val.shape

        coef_idx = (np.repeat(X_feat, n_fields, axis=-1), np.tile(X_field, n_fields))
        coef = self.latent_coef[coef_idx[0], coef_idx[1]]

        X_val_repeated = np.repeat(X_val, n_fields, axis=-1)[..., None]
        coef_val = (coef * X_val_repeated).reshape(n_samples, n_fields, n_fields, -1)

        triu = np.triu_indices(n_fields, k=1)
        kernel_ffm = np.sum(coef_val.transpose((0, 2, 1, 3))[:, triu[0], triu[1], :]
                            * coef_val[:, triu[0], triu[1], :], axis=(1, 2))

        return np.sign(kernel_ffm)
