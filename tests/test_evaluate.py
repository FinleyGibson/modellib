import unittest
from modellib.evaluate import leave_one_out_cv
from modellib.regression import LinearRegression
import numpy as np


class TestLeaveOneOutCV(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = LinearRegression()
        cls.x = [i.reshape(-1, 2) for i in np.random.randn(10, 2)]
        cls.y = [i.reshape(-1, 2) for i in np.arange(20).reshape(-1, 2)]
        pass

    def test_leave_one_out_cv(self):
        Y, Y_ = leave_one_out_cv(self.x, self.y, self.model)
        self.assertIsInstance(Y_, list)
        self.assertEqual(10, len(Y_))
        self.assertEqual(10, len(Y))
        self.assertEqual(2, Y_[0].shape[1])
        self.assertEqual(2, Y[0].shape[1])


if __name__ == '__main__':
    # to run all tests:
    unittest.main()

    # or to be specific:
    # unittest.main(TestSomething)
