import unittest


class Harness(unittest.TestCase):
    def test_holder(self):
        self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main()
