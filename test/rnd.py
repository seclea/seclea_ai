from typing import Union

import pandas as pd

from seclea_ai.lib.seclea_utils.object_management import Tracked

a: Union[Tracked, pd.DataFrame] = Tracked(pd.DataFrame())
