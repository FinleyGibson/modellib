import numpy as np
import numpy.typing as npt
from typing import Type, Any, List, Tuple
from modellib.base import Model
from tqdm import tqdm


def leave_one_out_cv(x: List[npt.NDArray[Any]],
                     y: List[npt.NDArray[Any]],
                     model: Type[Model]) -> Tuple[List[npt.NDArray[Any]],
                                                  List[npt.NDArray[Any]]]:
    """
    Perform leave-one-out cross-validation using the specified model.

    Args:
        x (List[NDArray]): List of input features for each participant.
        y (List[NDArray]): List of corresponding target values for each
        participant.
        model (Type[Model]): The machine learning model to be evaluated.

    Returns:
        Tuple[List[NDArray], List[NDArray]]: A tuple containing two lists.
            - The first list (Y) contains the true target values for each
            participant.
            - The second list (Y_) contains the predicted target values for
            each participant.

    Raises:
        AssertionError: If the lengths of input feature and target value lists
        (x and y) are not equal.

    Note:
        This function performs leave-one-out cross-validation by iteratively
        leaving out one participant for testing and using the remaining
        participants for training the specified model. It returns two lists
        containing true target values (Y) and predicted target values (Y_) for
        each participant.

    Example:
        >>> x_data = [...]  # List of input features for each participant
        >>> y_data = [...]  # List of corresponding target values for each
                            # participant
        >>> model_instance = YourModel()  # Instantiate your model
        >>> true_values, predicted_values = leave_one_out_cv(x_data,
                                                             y_data,
                                                             model_instance)
    """
    assert len(x) == len(y)
    n_participants = len(x)

    Y = []
    Y_ = []

    for i in tqdm(range(n_participants)):
        # split data to train and test
        x_te = x[i]
        y_te = y[i]
        x_tr = np.vstack(x[:i] + x[i+1:])
        y_tr = np.vstack(y[:i] + y[i+1:])

        # fit and evaluate model
        model.fit(x=x_tr, y=y_tr)
        y_ = model.predict(x_te)

        Y.append(y_te)
        Y_.append(y_)

    return Y, Y_


def random_undersample(xs: List[npt.NDArray[Any]],
                       ys: List[npt.NDArray[Any]]) -> \
        Tuple[npt.NDArray[Any], npt.NDArray[Any]]:

    assert len(xs) == len(ys)

    n_samples = min([xi.shape[0] for xi in xs])

    xs_out = []
    ys_out = []
    for xi, yi in zip(xs, ys):
        inds = np.random.permutation(n_samples)
        xs_out.append(xi[inds])
        ys_out.append(yi[inds])

    xs_out = np.vstack(xs_out)
    ys_out = np.vstack(ys_out)
    inds = np.random.permutation(xs_out.shape[0])

    return xs_out[inds], ys_out[inds]

