from collections import deque
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras


def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(output_size)
    else:
        Q_value = model.predict(tf.one_hot([state], input_size), verbose=0)
        return np.argmax(Q_value)

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array(
            [experience[field_index] for experience in batch]
        ) for field_index in range(5)
    ]
    return states, actions, rewards, next_states, dones

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(tf.one_hot(next_states, input_size), verbose=0)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards + (1 - dones) * discount_rate * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, output_size)
    with tf.GradientTape() as tape:
        all_Q_values = model(tf.one_hot(states, input_size))
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


env = gym.make("FrozenLake-v1", is_slippery=False)

input_size = env.observation_space.n
output_size = env.action_space.n

model = keras.models.Sequential([
    keras.layers.Dense(3, activation="relu", input_shape=[input_size]),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(output_size, activation="softmax"),
])

episodes = 1000
batch_size = 32
discount_rate = 0.99
optimizer = keras.optimizers.Adam(learning_rate=0.75)
loss_fn = keras.losses.CategoricalCrossentropy()
replay_memory = deque(maxlen=10000)

rewards = 0


for episode in range(episodes):
    obs = env.reset()    
    epsilon = max(1 - episode / (episodes*3/4.), 0.01)
    while True:
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        rewards += reward
        if done:
            break
    print(f"\rEpisode: {episode+1} / {episodes}, eps: {epsilon:.3f}, reward: {rewards}", end="")
    if episode > (episodes//10):
        training_step(batch_size)