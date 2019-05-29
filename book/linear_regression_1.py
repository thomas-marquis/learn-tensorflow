#Simple method
#%%
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

#%%
data_scaler = StandardScaler()

housing_data = data_scaler.fit_transform(housing.data)

n, p = housing_data.shape
housing_data_with_bias = np.c_[np.ones((n, 1)), housing_data]


#%%
X = tf.constant(housing_data_with_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

X_T = tf.transpose(X)
theta = tf.matrix_inverse(X_T @ X) @ X_T @ y

#%%
with tf.Session() as sess:
    theta_value = theta.eval()
    print(theta_value)

