from abc import ABC, abstractmethod
from typing import Optional, Any
import numpy.typing as npt


class Model(ABC):
    """
    Abstract class for ML models.
    """

    def __init__(self):
        self.x: Optional[npt.NDArray[float]] = None
        self.y: Optional[npt.NDArray[float]] = None
        self.n_data: int = 0
        self.dim_in: Optional[int] = None
        self.dim_out: Optional[int] = None
        self.model: Optional[object] = None
        self.model_params: dict[str: Any] = {}

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

    def predict(self, x_: npt.NDArray[float]) -> Any:
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
    def _predict(self, x_: npt.NDArray[float]) -> Any:
        """
        Make a prediction of $\overline{f}(\mathbf{x}')$ where $\overline{
        f(\cdot)}$ models the true process $f(\cdot).$
        This should be the specific prediction method for the model.
        :param x_: input $\mathbf{x}'$ be predicted
        :return:
        """
        pass


class Regression(Model, ABC):
    """
    Base class for regression models.
    All regressors should inherit from this.
    """

    @abstractmethod
    def _predict(self, x_: npt.NDArray[float]) -> float:
        pass

    def predict(self, x_: npt.NDArray[float]) -> float:
        return super(Regression, self).predict(x_)


