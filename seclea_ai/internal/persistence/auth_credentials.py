from peewee import CharField

from .db import BaseModel


class AuthCredentials(BaseModel):

    key = CharField()
    value = CharField()
