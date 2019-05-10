import pandas as pd
import numpy as np

from lab.lab2 import Lab2

#%%
lab2 = Lab2()

#%%

val = Lab2().do()

#%%
df = pd.DataFrame(
    [range(0, 20), range(20, 40)],
    index=["a", "b"]
)

#%%
df2 = pd.DataFrame()
df2["a"] = pd.Series(range(0, 20))
df2["b"] = pd.Series(range(20, 40))

#%%
df3: pd.DataFrame = lab2.do()

#%%
class A:

    def get_df(self):
        a = pd.DataFrame()
        a["a"] = pd.Series(range(0, 20))
        a["b"] = pd.Series(range(20, 40))

        return a

#%%
instance = A()
c = instance.get_df()

#%%
import os

def getRootPath() -> str:
    return os.getcwd().replace("\\", "/")

e = pd.DataFrame()
e["a"] = pd.Series(range(0, 20))
e["b"] = pd.Series(range(20, 40))

path = getRootPath() + "/data.csv"
with open(path, "w", encoding="utf-8") as file:
    e.to_csv(file, sep=",")
