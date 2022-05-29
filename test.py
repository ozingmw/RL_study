import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras


env = gym.make("FrozenLake-v1")

obs = env.reset()

print(tf.one_hot(np.array([0,1,2,3,4,5,6]), 16))

model = keras.models.Sequential([
    keras.layers.Dense(3, activation="relu", input_shape=[16]),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(4, activation="softmax"),
])

model.predict(np.zeros([1, 16]))