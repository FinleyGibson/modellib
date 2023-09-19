from modellib.base import Regression
import numpy.typing as npt
from multiprocessing import cpu_count
import xgboost as _xgb


class RandomForest(Regression):
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


class LinearRegression(Regression):
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


class XGBoost(Regression):
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
        return self.model.predict(x_)
