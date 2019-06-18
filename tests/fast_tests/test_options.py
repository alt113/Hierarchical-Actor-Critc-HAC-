import unittest
from hac.options import parse_options


class TestOptions(unittest.TestCase):
    """Tests for the parser method in options.py."""

    def test_parse_options(self):
        options = parse_options(args=['ur5'])
        self.assertFalse(options.retrain)
        self.assertFalse(options.show)
        self.assertFalse(options.train_only)
        self.assertFalse(options.verbose)
        self.assertEqual(options.layers, 1)
        self.assertEqual(options.time_scale, 10)

        options = parse_options(args=['ur5', '--retrain'])
        self.assertTrue(options.retrain)

        options = parse_options(args=['ur5', '--show'])
        self.assertTrue(options.show)

        options = parse_options(args=['ur5', '--train_only'])
        self.assertTrue(options.train_only)

        options = parse_options(args=['ur5', '--verbose'])
        self.assertTrue(options.verbose)

        options = parse_options(args=['ur5', '--layers', '2'])
        self.assertEqual(options.layers, 2)

        options = parse_options(args=['ur5', '--time_scale', '100'])
        self.assertEqual(options.time_scale, 100)


if __name__ == '__main__':
    unittest.main()
