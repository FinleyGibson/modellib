import unittest
from parameterized import parameterized

import numpy as np

from modellib.regression import Regression

REGRESSORS = Regression.__subclasses__()


def example_function(x: np.ndarray) -> float:
    out = (x ** 2 / 10 + -1 * np.cos(3 * x)).sum(axis=1)
    if out.ndim > 1:
        return out
    else:
        return out.reshape(-1, 1)


class TestModels(unittest.TestCase):
    x = None
    y = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.x = np.random.uniform(-10, 10, 50).reshape(25, 2)
        cls.y = example_function(cls.x)

    @parameterized.expand(REGRESSORS)
    def test_fit(self, Model):
        model = Model()

        model.fit(self.x, self.y)

        self.assertIsInstance(model.x, np.ndarray)
        self.assertIsInstance(model.y, np.ndarray)
        self.assertIsInstance(model.model, object)

        self.assertEqual(25, model.n_data)
        self.assertEqual(2, model.dim_in)
        self.assertEqual(1, model.dim_out)

        # should raise an error if a 1D np.array is provided as an arugment
        model = Model()
        self.assertRaises(AssertionError, model.fit, self.x, self.y.flatten())

    @parameterized.expand(REGRESSORS)
    def test_predict(self, Model):
        model = Model()
        model.fit(self.x, self.y)

        y_ = model.predict(self.x)
        self.assertEqual(self.y.shape, y_.shape)
        self.assertTrue(y_.dtype, float)


if __name__ == '__main__':
    # to run all tests:
    unittest.main()

    # or to be specific:
    # unittest.main(TestSomething)
