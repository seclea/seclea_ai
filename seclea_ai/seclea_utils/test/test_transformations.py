import unittest

from seclea_utils.core.transformations import decode_func, encode_func


class TestTransformations(unittest.TestCase):
    def test_encode_decode_function_no_args(self):
        def test_func(a):
            return 12 * 3 + a

        encoded = encode_func(test_func)
        decoded, args, kwargs = decode_func(encoded)
        self.assertEqual(decoded(2, *args, **kwargs), test_func(2))

    def test_encode_decode_function_args_only(self):
        def test_func(a, b):
            return a + b

        test_args = [1, -4]
        encoded = encode_func(test_func, args=test_args)
        decoded, args, kwargs = decode_func(encoded)
        self.assertEqual(decoded(*args, **kwargs), test_func(*test_args))

    def test_encode_decode_function_kwargs_only(self):
        def test_func(a, b):
            return a + (2 * b)

        test_kwargs = {"a": 1, "b": 3}

        encoded = encode_func(test_func, kwargs=test_kwargs)
        decoded, args, kwargs = decode_func(encoded)

        self.assertEqual(decoded(*args, **kwargs), test_func(**test_kwargs))  # add assertion here

    def test_encode_decode_function_args_kwargs(self):
        def test_func(a, b, c, d):
            return a - b + (c * d)

        test_args = [1, 4]
        test_kwargs = {"c": 3, "d": 5}

        encoded = encode_func(test_func, args=test_args, kwargs=test_kwargs)
        decoded, args, kwargs = decode_func(encoded)

        self.assertEqual(
            decoded(*args, **kwargs), test_func(*test_args, **test_kwargs)
        )  # add assertion here


if __name__ == "__main__":
    unittest.main()
