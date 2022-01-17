class DecompressionError(Exception):
    def __init__(self, message):
        super(DecompressionError, self).__init__(message)


class CompressionError(Exception):
    def __init__(self, message):
        super(CompressionError, self).__init__(message)
