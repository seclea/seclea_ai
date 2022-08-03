import os
import time
import unittest

import peewee
import responses

from seclea_ai.internal.api.api_interface import Api
from seclea_ai.internal.director import Director

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_path = os.path.join(base_dir, "")
print(folder_path)


class TestSecleaAIThreading(unittest.TestCase):
    @responses.activate
    def test_get_dataset_type(self):
        # ARRANGE
        # setup auth url
        responses.add(
            method=responses.POST,
            url="http://localhost:8010/api/token/verify/",
            status=200,
        )
        # set up director object which initialises threadpool.
        api = Api(  # nosec B106
            settings={"auth_url": "http://localhost:8010", "platform_url": "http://localhost:8000"},
            username="test",
            password="test",
        )
        director = Director(settings={}, api=api)

        # ACT
        # try and record some data and trigger an unhandled exception in the thread
        director.store_entity(
            {
                "entity": "dataset",
                "record_id": 100000,
                "dataset": "faked",
                "some": "non compliant data",
            }
        )
        # need to wait for exception to be raised and caught - this is not ideal but unavoidable for now.
        time.sleep(0.1)

        # ASSERT
        # make sure the exception propagates to user thread and gets thrown on second call
        with self.assertRaises(peewee.DoesNotExist):
            director.store_entity({"entity": "dataset"})


if __name__ == "__main__":
    unittest.main()
