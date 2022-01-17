# Original Credit https://stackoverflow.com/questions/23212435/permission-denied-to-write-to-my-temporary-file/63173312#63173312
# Extended by Roger Milroy

import os
import tempfile


class CustomNamedTemporaryFile:
    """
    This custom implementation is needed because of the following limitation of tempfile.NamedTemporaryFile:

    > Whether the name can be used to open the file a second time, while the named temporary file is still open,
    > varies across platforms (it can be so used on Unix; it cannot on Windows NT or later).
    """

    def __init__(self, mode="w+b", delete=True):
        self._mode = mode
        self._delete = delete
        self.name = None

    def __enter__(self):
        # Generate a random temporary file name
        self.name = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        # Ensure the file is created
        open(self.name, "x").close()
        # Open the file in the given mode
        self._tempFile = open(self.name, self._mode)
        return self._tempFile

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tempFile.close()
        if self._delete:
            os.remove(self.name)
