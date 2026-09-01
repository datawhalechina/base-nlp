import math
import unittest

import torch

from src.pos import PositionalEncoding


class PositionalEncodingTest(unittest.TestCase):
    def test_even_dimension_preserves_output_shape_and_finite_values(self):
        encoding = PositionalEncoding(dim=4, max_seq_len=6)
        output = encoding(torch.zeros(2, 6, 4))

        self.assertEqual(output.shape, (2, 6, 4))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_odd_dimension_preserves_output_shape_and_finite_values(self):
        encoding = PositionalEncoding(dim=3, max_seq_len=6)
        output = encoding(torch.zeros(2, 6, 3))

        self.assertEqual(output.shape, (2, 6, 3))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_odd_dimension_uses_sine_for_the_unpaired_even_index(self):
        encoding = PositionalEncoding(dim=3, max_seq_len=2)

        expected = math.sin(math.exp(2 * (-math.log(10000.0) / 3)))
        self.assertAlmostEqual(encoding.pe[0, 1, 2].item(), expected, places=6)


if __name__ == "__main__":
    unittest.main()
