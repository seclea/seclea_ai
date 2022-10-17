import json
from json import JSONDecodeError

from peewee import CharField


class JsonField(CharField):
    def db_value(self, value):
        return json.dumps(value)

    def python_value(self, value):
        try:
            value = json.loads(value)
        except JSONDecodeError:
            value = None
        finally:
            return value


class EnumField(CharField):
    def __init__(self, enum_class):
        super(EnumField, self).__init__()
        self.enum = enum_class

    def db_value(self, value):
        if not isinstance(value, self.enum):
            raise TypeError(f"Wrong type, must be {self.enum}")
        return super().adapt(value=value.value)

    def python_value(self, value):
        return self.enum(value)
