import unittest
from parameterized import parameterized, parameterized_class
from modellib.models import RandomForest
import numpy as np
from copy import deepcopy


def example_function(x: np.ndarray) -> float:
    return x ** 2 / 10 + -1 * np.cos(3 * x)


@parameterized_class([
    {"model_type": RandomForest, "args": {}},
])
class TestModels(unittest.TestCase):
    model_type = None
    args = {}

    @classmethod
    def setUpClass(cls) -> None:

        cls.x = np.random.uniform(-10, 10, 25).reshape(25, 1)
        cls.model = cls.model_type(**cls.args)

    def test_fit(self):
        y = example_function(self.x)
        model = deepcopy(self.model)

        model.fit(self.x, y)

        self.assertIsInstance(model.x, np.ndarray)
        self.assertIsInstance(model.y, np.ndarray)
        self.assertEqual(model.n_data, 25)

    def test_predict(self):
        y = example_function(self.x)
        model = deepcopy(self.model)

        model.fit(self.x, y)
        y_ = model.predict(self.x)

        self.assertEqual(y.shape, y_.shape)


if __name__ == '__main__':
    # to run all tests:
    unittest.main()

    # or to be specific:
    # unittest.main(TestSomething)
