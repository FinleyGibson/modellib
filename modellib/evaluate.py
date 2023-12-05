import numpy as np
import numpy.typing as npt
from typing import Type, Any, List
from modellib.base import Model
from tqdm import tqdm


def leave_one_out_cv(x: List[npt.NDArray[Any]],
                     y: List[npt.NDArray[Any]],
                     model: Type[Model]) -> List[npt.NDArray[Any]]:

    assert len(x) == len(y)
    n_participants = len(x)

    Y_ = []

    for i in tqdm(range(n_participants)):
        # split data to train and test
        x_te = x[i]

        x_tr = np.vstack(x[:i] + x[i+1:])
        y_tr = np.vstack(y[:i] + y[i+1:])

        # fit and evaluate model
        model.fit(x=x_tr, y=y_tr)
        y_ = model.predict(x_te)

        Y_.append(y_)

    return Y_



