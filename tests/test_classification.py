import unittest
from parameterized import parameterized

import numpy as np
import numpy.testing as nptest

from modellib.classification import Classification, LogisticRegression, XGBoost

CLASSIFIERS = Classification.__subclasses__()


def example_function(x: np.ndarray) -> float:
    return np.random.random_integers(0, 3, x.shape[0]).reshape(-1, 1)


class TestModels(unittest.TestCase):
    x = None
    y = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.x = np.random.uniform(-10, 10, 50).reshape(25, 2)
        cls.y = example_function(cls.x)

    @parameterized.expand(CLASSIFIERS)
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

    @parameterized.expand(CLASSIFIERS)
    def test_predict(self, Model):
        model = Model()
        model.fit(self.x, self.y)

        y_ = model.predict(self.x)
        self.assertEqual(self.y.shape, y_.shape)
        self.assertEqual(y_.dtype, int)

    @parameterized.expand(CLASSIFIERS)
    def test_predict_proba(self, Model):
        model = Model()
        model.fit(self.x, self.y)

        out = model.predict_proba(self.x, 1.0)
        nptest.assert_array_equal(out, np.ones_like(out))

        out = model.predict_proba(self.x, 0.0)
        nptest.assert_array_equal(out, np.zeros_like(out))


class TestXGBoost(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        x = np.random.randn(10, 3)
        y = np.random.randint(0, 2, (10, 1))
        cls.x = x
        cls.y = y
        cls.model = XGBoost()
        cls.model.fit(x, y)


class TestLogisticRegression(unittest.TestCase):
    """
    Tests specific to logistic regression classifier
    """
    @classmethod
    def setUpClass(cls) -> None:
        x = np.random.randn(10, 3)
        y = np.random.randint(0, 2, (10, 1))
        cls.x = x
        cls.y = y
        cls.model = LogisticRegression()
        cls.model.fit(x, y)

    def test_get_coefficients(self):
        coeffs = self.model.get_coefficients()
        self.assertIsInstance(coeffs, np.ndarray)
        self.assertEqual(coeffs.shape, (1, 3))

    def test_predict_proba(self):
        out = self.model.predict_proba(self.x, 0.5)

        out = self.model.predict_proba(self.x, 1.0)
        nptest.assert_array_equal(out, np.ones_like(out))

        out = self.model.predict_proba(self.x, 0.0)
        nptest.assert_array_equal(out, np.zeros_like(out))

if __name__ == '__main__':
    # to run all tests:
    unittest.main()

    # or to be specific:
    # unittest.main(TestSomething)
