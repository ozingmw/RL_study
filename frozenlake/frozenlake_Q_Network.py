import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras


env = gym.make("FrozenLake-v1")

input_size = env.observation_space.n
output_size = env.action_space.n

learning_rate = 0.1

model = keras.models.Sequential([
    keras.layers.Dense(3, activation="relu", input_shape=[input_size]),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(output_size),
])


# X = tf.placeholder(shape=[1, input_size], dtype=tf.float32)
# w = tf.Variable(tf.random.uniform([input_size, output_size], 0, 0.01))
# y_pred = tf.matmul(X, w)
# y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)
# loss = tf.reduce_sum(tf.square(y - y_pred))
# train = tf.train


for episode in episodes:
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            break
    train_step(inputs)

@tf.function
def train_step(inputs):
    batch_data, labels = inputs
    with tf.GradientTape() as tape:
        predictions = model(batch_data, training=True)
        loss = keras.losses.CategoricalCrossentropy(reduction=keras.losses.Reduction.NONE)(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))