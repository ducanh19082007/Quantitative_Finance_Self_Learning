import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def train_test_split_sc(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    n_test = int(len(X) * test_size)
    return X[indices[:-n_test]], X[indices[-n_test:]], y[indices[:-n_test]], y[indices[-n_test:]]


def _standardize_1d(X, y):
    X_mean, X_std = X.mean(), X.std() + 1e-8
    y_mean, y_std = y.mean(), y.std() + 1e-8
    return (X - X_mean) / X_std, (y - y_mean) / y_std, X_mean, X_std, y_mean, y_std


def _unscale_1d(slope_s, X_mean, X_std, y_mean, y_std):
    """
    math equation for this part:
    slope = slope_s * (y_std / X_std)
    bias  = y_mean - slope * X_mean
    """
    slope = slope_s * (y_std / X_std)
    bias  = y_mean - slope * X_mean
    return slope, bias


def _standardize_nd(X, y):
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-8
    y_mean = y.mean()
    y_std  = y.std() + 1e-8
    return (X - X_mean) / X_std, (y - y_mean) / y_std, X_mean, X_std, y_mean, y_std


def _unscale_nd(w_s, X_mean, X_std, y_mean, y_std):
    w    = w_s * (y_std / X_std)
    bias = y_mean - X_mean @ w
    return w, bias


# ---------------------------------------------------------------------------
# Plain linear regression (gradient descent)
# ---------------------------------------------------------------------------

class Linear_Regression_Sc:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.slope = 0.0
        self.bias  = 0.0

    def fit(self, X, y):
        # ravel() fixes silent (n,1) broadcasting bug:
        # ys-(Xs*w+b) would become (n,n) if X is a column vector, blowing up gradients.
        X, y = np.asarray(X).ravel(), np.asarray(y).ravel()
        Xs, ys, X_mean, X_std, y_mean, y_std = _standardize_1d(X, y)

        slope_s, bias_s = 0.0, 0.0
        n = len(Xs)
        for _ in range(self.epochs):
            residual = ys - (Xs * slope_s + bias_s)
            slope_s -= self.learning_rate * (-2 / n) * np.sum(Xs * residual)
            bias_s  -= self.learning_rate * (-2 / n) * np.sum(residual)

        self.slope, self.bias = _unscale_1d(slope_s, X_mean, X_std, y_mean, y_std)

    def predict(self, X):
        return self.slope * np.asarray(X).ravel() + self.bias

    def show(self):
        print(f"Slope: {self.slope}, Bias: {self.bias}")


# ---------------------------------------------------------------------------
# Lasso (L1) regression
# ---------------------------------------------------------------------------

class Lasso_Linear_Regression_Sc:
    def __init__(self, learning_rate=0.01, epochs=1000, lambda_=0.1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_ = lambda_
        self.slope = 0.0
        self.bias  = 0.0

    def fit(self, X, y):
        X, y = np.asarray(X).ravel(), np.asarray(y).ravel()
        Xs, ys, X_mean, X_std, y_mean, y_std = _standardize_1d(X, y)

        slope_s, bias_s = 0.0, 0.0
        n = len(Xs)
        for _ in range(self.epochs):
            residual = ys - (Xs * slope_s + bias_s)
            slope_s -= self.learning_rate * ((-2 / n) * np.sum(Xs * residual) + self.lambda_ * np.sign(slope_s))
            bias_s  -= self.learning_rate * (-2 / n) * np.sum(residual)

        self.slope, self.bias = _unscale_1d(slope_s, X_mean, X_std, y_mean, y_std)

    def predict(self, X):
        return self.slope * np.asarray(X).ravel() + self.bias

    def show(self):
        print(f"Slope: {self.slope}, Bias: {self.bias}")


# ---------------------------------------------------------------------------
# Ridge (L2) regression
# ---------------------------------------------------------------------------

class Ridge_Linear_Regression_Sc:
    def __init__(self, learning_rate=0.01, epochs=1000, lambda_=0.1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_ = lambda_
        self.slope = 0.0
        self.bias  = 0.0

    def fit(self, X, y):
        X, y = np.asarray(X).ravel(), np.asarray(y).ravel()
        Xs, ys, X_mean, X_std, y_mean, y_std = _standardize_1d(X, y)

        slope_s, bias_s = 0.0, 0.0
        n = len(Xs)
        for _ in range(self.epochs):
            residual = ys - (Xs * slope_s + bias_s)
            slope_s -= self.learning_rate * ((-2 / n) * np.sum(Xs * residual) + 2 * self.lambda_ * slope_s)
            bias_s  -= self.learning_rate * (-2 / n) * np.sum(residual)

        self.slope, self.bias = _unscale_1d(slope_s, X_mean, X_std, y_mean, y_std)

    def predict(self, X):
        return self.slope * np.asarray(X).ravel() + self.bias

    def show(self):
        print(f"Slope: {self.slope}, Bias: {self.bias}")


# ---------------------------------------------------------------------------
# Ridge with cross-validated lambda (closed-form inner solver)
# ---------------------------------------------------------------------------

class RidgeCV_Linear_Regression_Sc:
    def __init__(self, learning_rate=0.01, epochs=1000, lambda_values=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_values = lambda_values if lambda_values is not None else [0.1, 1, 10]
        self.slope = None
        self.bias  = 0.0

    def _inner_ridge(self, X, y, lambda_):
        I = np.eye(X.shape[1])
        return np.linalg.solve(X.T @ X + lambda_ * I, X.T @ y)

    def _ridge_cv(self, X, y, k_folds=5):
        folds = np.array_split(np.random.permutation(X.shape[0]), k_folds)
        best_lambda, min_err, cv_log = None, float("inf"), {}
        for lam in self.lambda_values:
            mse_list = []
            for j in range(k_folds):
                val = folds[j]
                train = np.hstack([folds[i] for i in range(k_folds) if i != j])
                w   = self._inner_ridge(X[train], y[train], lam)
                mse = np.mean((y[val] - X[val] @ w) ** 2)
                mse_list.append(mse)
            avg = np.mean(mse_list)
            cv_log[lam] = avg
            if avg < min_err:
                min_err, best_lambda = avg, lam
        return best_lambda, cv_log

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Xs, ys, X_mean, X_std, y_mean, y_std = _standardize_nd(X, y)
        best_lambda, cv_log = self._ridge_cv(Xs, ys)
        print(f"Best lambda: {best_lambda}")
        print(f"CV Error Log: {cv_log}")
        w_s = self._inner_ridge(Xs, ys, best_lambda)
        self.slope, self.bias = _unscale_nd(w_s, X_mean, X_std, y_mean, y_std)

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.slope + self.bias

    def show(self):
        print(f"Slope: {self.slope}, Bias: {self.bias}")


# ---------------------------------------------------------------------------
# Lasso with cross-validated lambda (gradient-descent inner solver)
# ---------------------------------------------------------------------------

class LassoCV_Linear_Regression_Sc:
    def __init__(self, learning_rate=0.01, epochs=1000, lambda_values=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_values = lambda_values if lambda_values is not None else [0.1, 1, 10]
        self.slope = None
        self.bias  = 0.0

    def _inner_lasso(self, X, y, lambda_):
        w, n = np.zeros(X.shape[1]), len(X)
        for _ in range(self.epochs):
            residual = y - X @ w
            w -= self.learning_rate * ((-2 / n) * (X.T @ residual) + lambda_ * np.sign(w))
        return w

    def _lasso_cv(self, X, y, k_folds=5):
        folds = np.array_split(np.random.permutation(X.shape[0]), k_folds)
        best_lambda, min_err, cv_log = None, float("inf"), {}
        for lam in self.lambda_values:
            mse_list = []
            for j in range(k_folds):
                val = folds[j]
                train = np.hstack([folds[i] for i in range(k_folds) if i != j])
                w   = self._inner_lasso(X[train], y[train], lam)
                mse = np.mean((y[val] - X[val] @ w) ** 2)
                mse_list.append(mse)
            avg = np.mean(mse_list)
            cv_log[lam] = avg
            if avg < min_err:
                min_err, best_lambda = avg, lam
        return best_lambda, cv_log

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Xs, ys, X_mean, X_std, y_mean, y_std = _standardize_nd(X, y)
        best_lambda, cv_log = self._lasso_cv(Xs, ys)
        print(f"Best lambda: {best_lambda}")
        print(f"CV Error Log: {cv_log}")
        w_s = self._inner_lasso(Xs, ys, best_lambda)
        self.slope, self.bias = _unscale_nd(w_s, X_mean, X_std, y_mean, y_std)

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.slope + self.bias

    def show(self):
        print(f"Slope: {self.slope}, Bias: {self.bias}")