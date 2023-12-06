import unittest
from modellib.evaluate import leave_one_out_cv, random_undersample
from modellib.regression import LinearRegression
import numpy as np
import numpy.testing as nptest


class TestLeaveOneOutCV(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = LinearRegression()
        cls.x = [i.reshape(-1, 2) for i in np.random.randn(10, 2)]
        cls.y = [i.reshape(-1, 2) for i in np.arange(20).reshape(-1, 2)]

    def test_leave_one_out_cv(self):
        Y, Y_ = leave_one_out_cv(self.x, self.y, self.model)
        self.assertIsInstance(Y_, list)
        self.assertEqual(10, len(Y_))
        self.assertEqual(10, len(Y))
        self.assertEqual(2, Y_[0].shape[1])
        self.assertEqual(2, Y[0].shape[1])


class TestRandomUndersample(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.xs = [np.ones((i*5, 10))*i for i in range(1, 5)]
        cls.ys = [np.ones((i*5, 1))*i for i in range(1, 5)]

    def test_random_undersample(self):
        xs_out, ys_out = random_undersample(xs=self.xs, ys=self.ys)

        self.assertEqual((20, 10), xs_out.shape)
        self.assertEqual((20, 1), ys_out.shape)

        nptest.assert_array_equal(np.arange(1, 5), np.unique(ys_out))


if __name__ == '__main__':
    # to run all tests:
    unittest.main()

    # or to be specific:
    # unittest.main(TestSomething)
