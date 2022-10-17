import json
from json import JSONDecodeError

from peewee import Field


class JsonField(Field):
    def db_value(self, value):
        return json.dumps(value)

    def python_value(self, value):
        try:
            value = json.loads(value)
        except JSONDecodeError:
            value = None
        finally:
            return value
