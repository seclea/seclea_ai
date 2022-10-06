from pathlib import Path

from sqlitedict import SqliteDict  # nosec


class Storage:
    """Simple storage supporting pickledb for now."""

    def __init__(self, root: str = None, db_name="default.db"):
        """
        @param root: root path of db default is Path.home()/.seclea/
        @param db_name: name of database
        """
        if root is None:
            root = f"{Path.home()}/.seclea/"
        self._path = f"{root}/{db_name}"
        Path(root).mkdir(parents=True, exist_ok=True)
        # TODO this is unclosed during testing...
        self.db = SqliteDict(self._path)

    @property
    def path(self):
        return self._path

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()

    @path.setter
    def path(self, path: str):
        """
        :param path:
        :return:
        """
        path_root = path.split("/")[-1].join("/")
        Path(path_root).mkdir(parents=True, exist_ok=True)
        self._path = path
        # TODO this is unclosed during testing...
        self.db.close()
        self.db = SqliteDict(self._path)

    def write(self, key, val):
        self.db[key] = val
        self.db.commit()

    def get(self, key):
        return self.db.get(key)
