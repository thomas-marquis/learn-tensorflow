# from https://colab.research.google.com/notebooks/mlcc/intro_to_pandas.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=pandas-colab&hl=fr#scrollTo=8UngIdVhz8C0
from __future__ import print_function

import pandas as pd

pd.__version__

#%%
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

df = pd.DataFrame({"cityName": city_names, "population": population})

#%%
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe.describe()

#%%
california_housing_dataframe.hist('housing_median_age')

#%%
cities = pd.DataFrame({"cityName": city_names, "population": population})
print(type(cities))
##cities["cityName"]
cities.cityName

#%%
cities[0:2]

#%%
cities["cityName"][0:2]

#%%
population / 1000

#%%
import numpy as np

np.log(population)

#%%
pop_mask = population.apply(lambda pop: pop > 1000000)
print(pop_mask)
population[pop_mask]

#%%
population[population > 1000000]

#%%
cities['area'] = pd.Series([46.87, 176.53, 97.92])
cities["density"] = cities.population / cities.area
cities.head()

#%% EXERCICE 1
cities["saint"] = cities.cityName.apply(lambda name: name.startswith("San"))
cities.head()

#%% EXERCICE 1 CORR
cities["wide_and_san"] = (cities.area > 50) & (cities.cityName.apply(lambda name: name.startswith("San")))
cities

#%%
print(cities)
cities.reindex([0, 1, 2])
print(cities)
cities.reindex([2, 0, 1])
cities

#%%
cities.reindex(np.random.permutation(cities.index))

#%% EXERCICE 2
cities.reindex([0,3,2, 44, 1])
