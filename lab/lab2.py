import pandas as pd
import numpy as np


class Lab2:

    def do(self) -> pd.DataFrame:
        # df2 = pd.DataFrame()
        # df2["a"] = pd.Series(range(0, 20))
        # df2["b"] = pd.Series(range(20, 40))

        # return df2

        return pd.DataFrame(
            [range(0, 20), range(20, 40)],
            index=["a", "b"]
        )

e = pd.DataFrame()
e["a"] = pd.Series(range(0, 20))
e["b"] = pd.Series(range(20, 40))

with open("/out/data.csv", "w", encoding="utf-8") as file:
    e.to_csv(file, sep=",")

