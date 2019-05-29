#Gradiant method
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
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(housing_data_with_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape((-1, 1)), dtype=tf.float32, name="y")

theta = tf.Variable(tf.random_uniform([p + 1, 1], -1.0, 1.0), dtype=tf.float32, name="theta")

#%%
y_pred = tf.matmul(X, theta, name="prediction")
pred_error = y_pred - y
mse = tf.reduce_mean(tf.square(pred_error), name="mse")
# gradient = 2 / n * tf.transpose(X) @ pred_error #without auto diff
gradient = tf.gradients(mse, [theta])[0] # with auto diff => perfs ++
training_op = tf.assign(theta, theta - learning_rate * gradient)

init = tf.global_variables_initializer()

#%%
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("epoch {}, mse={}".format(epoch, mse.eval()))
        sess.run(training_op)
    best_theta = theta.eval()
    print("final theta={}".format(best_theta))
