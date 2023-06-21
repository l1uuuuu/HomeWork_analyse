import numpy as np

class NaiveBayes:
    def __init__(self):
        self.prior = None
        self.likelihood = None
        self.classes = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # 计算先验概率和似然
        self.prior = np.zeros(n_classes)
        self.likelihood = np.zeros((n_classes, n_features))
        for c in self.classes:
            X_c = X[y == c]
            self.prior[c] = X_c.shape[0] / float(n_samples)
            self.likelihood[c, :] = (X_c.sum(axis=0) + 1) / (X_c.sum() + n_features)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        posteriors = []
        for c in self.classes:
            prior_c = np.log(self.prior[c])
            likelihood_c = np.sum(np.log(self.likelihood[c, :]) * x)
            posterior_c = prior_c + likelihood_c
            posteriors.append(posterior_c)
        return self.classes[np.argmax(posteriors)]
