import pandas as pd
from wrapt.wrappers import ObjectProxy
import datetime
from seclea_ai import SecleaAI
import uuid


class Tracked:
    start_timestamp = datetime.datetime.now()
    password = "asdf"  # nosec
    username = "onespanadmin"  # nosec
    organization = "Onespan"
    project = f"test-project-{uuid.uuid4()}"
    portal_url = "http://localhost:8000"
    auth_url = "http://localhost:8010"
    controller = SecleaAI(
        project_name=project,
        organization=organization,
        platform_url=portal_url,
        auth_url=auth_url,
        username=username,
        password=password,
    )

    def __init__(self, wrapped, controler=None):
        print(type(wrapped))
        # super(Tracked, self).__init__(wrapped)
        if controler is not None:
            self.controller = controler

    # def __call__(self, *args, **kwargs):
    #     print('entering', self.__wrapped__.__name__)
    #     try:
    #         return self.__wrapped__(*args, **kwargs)
    #     finally:
    #         print('exiting', self.__wrapped__.__name__)
b=Tracked(pd.DataFrame())
a=Tracked(pd.DataFrame,controler='hello')
Tracked.controller='goodbye'
print(type(a))
print(a.__class__.controller)
print(a.controller)
#
# df1: pd.DataFrame = Tracked(pd.DataFrame(
#     {
#         "A": ["A0", "A1", "A2", "A3"],
#         "B": ["B0", "B1", "B2", "B3"],
#         "C": ["C0", "C1", "C2", "C3"],
#         "D": ["D0", "D1", "D2", "D3"],
#     },
#     index=[0, 1, 2, 3],
# )
# )
# df2 = pd.DataFrame(
#     {
#         "A": ["A4", "A5", "A6", "A7"],
#         "B": ["B4", "B5", "B6", "B7"],
#         "C": ["C4", "C5", "C6", "C7"],
#         "D": ["D4", "D5", "D6", "D7"],
#     },
#     index=[4, 5, 6, 7],
# )
#
# df3 = pd.DataFrame(
#     {
#         "A": ["A8", "A9", "A10", "A11"],
#         "B": ["B8", "B9", "B10", "B11"],
#         "C": ["C8", "C9", "C10", "C11"],
#         "D": ["D8", "D9", "D10", "D11"],
#     },
#     index=[8, 9, 10, 11],
# )
# a = Tracked(0)
# b = Tracked(1)
# print(type(a))
# print(a + b)
# print(a * 1)
# print(sum([1, a, 2, a]))
# print(df1.join(df2))
