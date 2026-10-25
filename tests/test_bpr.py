import unittest

from src import config


class BPRTests(unittest.TestCase):
    def test_fixed_training_settings(self) -> None:
        self.assertEqual(config.BPR_DIM, 64)
        self.assertEqual(config.BPR_EPOCHS, 20)
        self.assertEqual(config.BPR_NEGATIVES, 1)
        self.assertEqual(config.BPR_LEARNING_RATE, 0.001)
        self.assertEqual(config.BPR_L2, 0.0001)
        self.assertEqual(config.BPR_INIT_STD, 0.1)
        self.assertEqual(config.LIGHTGCN_LAYERS, 2)


if __name__ == "__main__":
    unittest.main()
