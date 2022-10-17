from enum import Enum
from unittest import TestCase
from peewee import SqliteDatabase, Model

from seclea_ai.internal.models.fields import JsonField, EnumField


class BaseFieldTest(TestCase):
    def setUp(self) -> None:
        self.db = SqliteDatabase(database=":memory:")
        self.db.connect()

    def tearDown(self) -> None:
        # this destroys the db.
        self.db.close()


class TestJsonField(BaseFieldTest):
    # TODO add NaN encoding test.

    def test_set_and_get_dict(self):
        # ARRANGE
        class TestModel(Model):
            json_field = JsonField()

            class Meta:
                database = self.db

        test_value = {"testing": 1}

        with self.db.atomic() as txn:
            self.db.create_tables(models=[TestModel])
            txn.commit()

            # ACT
            TestModel.create(json_field=test_value)
            txn.commit()

        # there should only be one entity in the db as created new everytime
        # note the index starts from 1 (peewee/sqlite decision)
        test_model = TestModel.get_or_none(1)

        # ASSERT
        self.assertIsNotNone(test_model, msg="TestModel from db is None")
        self.assertEqual(test_model.json_field, test_value, msg="From db doesn't equal input to db")

    def test_set_and_get_list(self):
        # ARRANGE
        class TestModel(Model):
            json_field = JsonField()

            class Meta:
                database = self.db

        # NOTE: no error on non string key in dict (not json compliant) just silently converts.
        # TODO should we add more validation? This is json.dumps behaviour not JsonField specific.
        test_value = [{"testing": 1}, 23, {"a": {"12": "hello"}}]

        with self.db.atomic() as txn:
            self.db.create_tables(models=[TestModel])
            txn.commit()

            # ACT
            TestModel.create(json_field=test_value)
            txn.commit()

        # there should only be one entity in the db as created new everytime
        # note the index starts from 1 (peewee/sqlite decision)
        test_model = TestModel.get_or_none(1)

        # ASSERT
        self.assertIsNotNone(test_model, msg="TestModel from db is None")
        self.assertEqual(test_model.json_field, test_value, msg="From db doesn't equal input to db")


class TestEnumField(BaseFieldTest):
    def test_set_and_get(self):
        # ARRANGE

        class TestEnum(Enum):
            ALPHA = "alpha"
            BETA = "beta"

        class TestModel(Model):
            enum_field = EnumField(TestEnum)

            class Meta:
                database = self.db

        test_value = TestEnum.ALPHA

        with self.db.atomic() as txn:
            self.db.create_tables(models=[TestModel])
            txn.commit()

            # ACT
            TestModel.create(enum_field=test_value)
            txn.commit()

        # there should only be one entity in the db as created new everytime
        # note the index starts from 1 (peewee/sqlite decision)
        test_model = TestModel.get_or_none(1)

        # ASSERT
        self.assertIsNotNone(test_model, msg="TestModel from db is None")
        self.assertEqual(test_model.enum_field, test_value, msg="From db doesn't equal input to db")
