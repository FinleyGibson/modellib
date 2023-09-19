from abc import ABCMeta, abstractmethod

import numpy as np
import numpy.typing as npt
from typing import Optional
from multiprocessing import cpu_count
import xgboost as _xgb
from tensorflow.keras import layers, models


class Regressor(metaclass=ABCMeta):

    def __init__(self):
        self.x: Optional[npt.NDArray[float]] = None
        self.y: Optional[npt.NDArray[float]] = None
        self.n_data: int = 0
        self.dim_in: Optional[int] = None
        self.dim_out: Optional[int] = None
        self.model: Optional[object] = None
        self.model_params: dict = {}

    def fit(self, x: npt.NDArray[float], y: npt.NDArray[float]):
        """
        Fits the model to the inputs $\mathbf{x}$ and $\mathbf{y}$. wraps
        self._fit which should be defined for each Regressor implementation.
        :param x: model inputs $\mathbf{x}$
        :param y: model inputs $\mathbf{y}$
        :return:
        """
        assert x.ndim == 2, f"inputs x, passed to fit should be " \
                            "two-dimensional. has shape {x.shape}"
        assert y.ndim == 2, f"inputs x, passed to fit should be " \
                            "two-dimensional. has shape {y.shape}"
        self.x = x
        self.y = y
        self.dim_in = x.shape[1]
        self.dim_out = y.shape[1]
        assert x.shape[0] == y.shape[0]
        self.n_data = x.shape[0]

        self._fit(x=self.x, y=self.y)

    @abstractmethod
    def _fit(self, x: npt.NDArray[float], y: npt.NDArray[float]):
        """
        Fits the specific model to the inputs $\mathbf{x}$ and $\mathbf{y}$.
        This should be specifically defined for each regression model.
        :param x: model inputs $\mathbf{x}$ shape (n_data, n_param)
        :param y: model inputs $\mathbf{y}$ shape (n_data, 1)
        :return:
        """
        pass

    def predict(self, x_: npt.NDArray[float]) -> npt.NDArray[float]:
        """
        Make a prediction of $\overline{f}(\mathbf{x}')$ where $\overline{
        f(\cdot)}$ models the true process $f(\cdot).$
        This should be specifically defined for each regression model.
        :param x_: input $\mathbf{x}'$ be predicted
        :return:
            prediction $\overline{f}(\mathbf{x}')$
        """
        assert x_.ndim == 2, f"predict should be passed a two " \
                             f"dimensional vector. x_.shape = {x_.shape}"
        assert x_.shape[1] == self.dim_in, "dimension mismatch on value " \
                                           "passed to Regressor.predict. " \
                                           "Expected two dimensional vector," \
                                           f" got shape {x_.shape}"

        out = self._predict(x_=x_).reshape(-1, self.dim_out)
        assert out.ndim == 2, "prediction returned by _predict should be " \
                              "two-dimensional."
        assert out.shape[0] == x_.shape[0], f"prediction returned by " \
                                            f"_predict should be of shape:" \
                                            f"{(x_.shape[0], self.dim_in)} " \
                                            f"but is instead shape: " \
                                            f"{out.shape}"
        assert out.shape[1] == self.dim_out, f"prediction returned by " \
                                             f"_predict should be of shape:" \
                                             f"{(x_.shape[0], self.dim_in)} " \
                                             f"but is instead shape: " \
                                             f"{out.shape}"
        return out

    @abstractmethod
    def _predict(self, x_: npt.NDArray[float]):
        """
        Make a prediction of $\overline{f}(\mathbf{x}')$ where $\overline{
        f(\cdot)}$ models the true process $f(\cdot).$
        This should be the specific prediction method for the model.
        :param x_: input $\mathbf{x}'$ be predicted
        :return:
        """
        pass


class RandomForest(Regressor):
    """
    Wrapper for sklearn's implementation of a RandomForestRegressor
    """
    from sklearn.ensemble import RandomForestRegressor as \
        _RandomForestRegressor

    def __init__(self, **kwargs):
        super(RandomForest, self).__init__()
        self.model_parameters = kwargs.copy()
        self.model = self._RandomForestRegressor(**kwargs)

    def _fit(self, x: npt.NDArray[float], y: npt.NDArray[float]):
        if self.dim_out > 1:
            self.model.fit(x, y)
        else:
            self.model.fit(x, y.reshape(-1))

    def _predict(self, x_: npt.NDArray[float]) -> npt.NDArray[float]:
        return self.model.predict(x_)


class LinearRegression(Regressor):
    """
    Wrapper for sklearn's implementation of a LinearRegression
    """
    from sklearn.linear_model import LinearRegression as _LinearRegression
    
    def __init__(self, **kwargs):
        super(LinearRegression, self).__init__()
        self.model_parameters = kwargs.copy()
        self.model = self._LinearRegression(**kwargs)

    def _fit(self, x: npt.NDArray[float], y: npt.NDArray[float]):
        if self.dim_out > 1:
            self.model.fit(x, y)
        else:
            self.model.fit(x, y.reshape(-1))

    def _predict(self, x_: npt.NDArray[float]) -> npt.NDArray[float]:
        return self.model.predict(x_)


class XGBoost(Regressor):
    """
    Wrapper for xgboost regressor
    """
    
    def __init__(self, **kwargs):
        super(XGBoost, self).__init__()
        default_kwargs = {"n_jobs": cpu_count(),
                          "n_estimators": 100}
        default_kwargs.update(kwargs)
        self.model = _xgb.XGBRegressor(**default_kwargs)
        self.model_parameters = default_kwargs

    def _fit(self, x: npt.NDArray[float], y: npt.NDArray[float]):
        self.model.fit(x, y)

    def _predict(self, x_: npt.NDArray[float]):
        return self.model.predict(x_=x_)

class Cnn(Regressor):

    def __int__(self):
        self.model = models.Sequential()
        model.add(layers.C)