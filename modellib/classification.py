from modellib.base import Model
from abc import ABC, abstractmethod
import numpy.typing as npt


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


