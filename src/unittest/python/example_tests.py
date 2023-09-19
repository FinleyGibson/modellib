import unittest
from example import function


class TestSomething(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """
        Code here runs once at the start of all tests
        """
        pass

    def setUp(self) -> None:
        """
        code here runs once at the start of each test

        """
        pass

    def test_example_one(self):
        self.assertEqual(0, function())


if __name__ == '__main__':
    # to run all tests:
    unittest.main()

    # or to be specific:
    # unittest.main(TestSomething)
