from modellib.base import Model
from abc import ABC, abstractmethod
import numpy.typing as npt
import xgboost as _xgb
from typing import Optional


class Classification(Model, ABC):
    """
    Base class for classification models.
    All classifiers should inherit from this.
    """
    def fit(self, x: npt.NDArray[float], y: npt.NDArray[int]):
        return super(Classification, self).fit(x, y)

    @abstractmethod
    def _fit(self, x: npt.NDArray[float], y: npt.NDArray[int]):
        pass

    @abstractmethod
    def _predict(self, x_: npt.NDArray[float]) -> npt.NDArray[int]:
        pass

    def predict(self, x_: npt.NDArray[float]) -> npt.NDArray[int]:
        return super(Classification, self).predict(x_)


class LogisticRegression(Classification):

    from sklearn.linear_model import LogisticRegression as _LogisticRegression

    def __init__(self, **kwargs):
        super(LogisticRegression, self).__init__()
        self.model_parameters = kwargs.copy()
        self.model = self._LogisticRegression(**kwargs)

    def _fit(self, x: npt.NDArray[float], y: npt.NDArray[int]):
        if self.dim_out > 1:
            self.model.fit(x, y)
        else:
            self.model.fit(x, y.reshape(-1))

    def _predict(self, x_: npt.NDArray[float]) -> npt.NDArray[int]:
        return self.model.predict(x_)

    def predict_proba(self, x_: npt.NDArray[float], thresh: Optional[float]) \
            -> \
            npt.NDArray[int]:

        probs = self.model.predict_proba(x_)
        if thresh is not None:
            return (probs[:, 0]<thresh).astype(int)
        else:
            return probs

    def get_coefficients(self) -> npt.NDArray[float]:
        return self.model.coef_


class RandomForest(Classification):

    from sklearn.ensemble import RandomForestClassifier as _RFClassifier

    def __init__(self, **kwargs):
        super(RandomForest, self).__init__()
        self.model_parameters = kwargs.copy()
        self.model = self._RFClassifier(**kwargs)

    def _fit(self, x: npt.NDArray[float], y: npt.NDArray[int]):
        if self.dim_out > 1:
            self.model.fit(x, y)
        else:
            self.model.fit(x, y.reshape(-1))

    def _predict(self, x_: npt.NDArray[float]) -> npt.NDArray[int]:
        return self.model.predict(x_)

    def predict_proba(self, x_: npt.NDArray[float], thresh: Optional[float]) \
            -> npt.NDArray[int]:

        probs = self.model.predict_proba(x_)
        if thresh is not None:
            return (probs[:, 0]<thresh).astype(int)
        else:
            return probs


class XGBoost(Classification):

    def __init__(self, **kwargs):
        super(XGBoost, self).__init__()
        default_kwargs = {"objective": 'binary:logistic',
                          "n_estimators": 100}
        default_kwargs.update(kwargs)
        self.model = _xgb.XGBClassifier(**default_kwargs)
        self.model_parameters = default_kwargs

    def _fit(self, x: npt.NDArray[float], y: npt.NDArray[float]):
        self.model.fit(x, y)

    def _predict(self, x_: npt.NDArray[float]):
        return self.model.predict(x_)

    def predict_proba(self, x_: npt.NDArray[float], thresh: Optional[float]) \
            -> npt.NDArray[int]:

        probs = self.model.predict_proba(x_)
        if thresh is not None:
            return (probs[:, 0]<thresh).astype(int)
        else:
            return probs
