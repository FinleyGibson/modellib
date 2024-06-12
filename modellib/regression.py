from modellib.base import Regression
import numpy.typing as npt
from multiprocessing import cpu_count


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

    def _fit(self, x: npt.NDArray, y: npt.NDArray):
        if self.dim_out > 1:
            self.model.fit(x, y)
        else:
            self.model.fit(x, y.reshape(-1))

    def _predict(self, x_: npt.NDArray) -> npt.NDArray:
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

    def _fit(self, x: npt.NDArray, y: npt.NDArray):
        if self.dim_out > 1:
            self.model.fit(x, y)
        else:
            self.model.fit(x, y.reshape(-1))

    def _predict(self, x_: npt.NDArray) -> npt.NDArray:
        return self.model.predict(x_)


class XGBoost(Regression):
    """
    Wrapper for xgboost regressor
    """

    import xgboost as _xgb
    
    def __init__(self, **kwargs):
        super(XGBoost, self).__init__()
        default_kwargs = {"n_jobs": cpu_count(),
                          "n_estimators": 100}
        default_kwargs.update(kwargs)
        self.model = self._xgb.XGBRegressor(**default_kwargs)
        self.model_parameters = default_kwargs

    def _fit(self, x: npt.NDArray, y: npt.NDArray):
        self.model.fit(x, y)

    def _predict(self, x_: npt.NDArray):
        return self.model.predict(x_)


class SVM(Regression):
    """_summary_
        Wrapper for sklearns Support Vector Machine (SVM)
    """

    from sklearn import svm as _svm

    def __init__(self, **kwargs):
        super(SVM, self).__init__()
        
        default_kwargs = {
            "kernel": "rbf"
            } 
        default_kwargs.update(kwargs)

        self.model = self._svm.SVR(**kwargs)

    def _fit(self, x: npt.NDArray, y: npt.NDArray):
        if self.dim_out > 1:
            self.model.fit(x, y)
        else:
            self.model.fit(x, y.reshape(-1))

    def _predict(self, x_: npt.NDArray):
        return self.model.predict(x_)