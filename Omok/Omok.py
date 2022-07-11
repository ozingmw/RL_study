from collections import deque
import time
import gym
import numpy as np
import gym_omok.envs.__init__

import tensorflow as tf
from tensorflow import keras


def to_one_hot(array):
    return tf.one_hot(array, input_size)

def nargmax(array):
    epsilon = 1e-6
    array += np.random.rand(*array.shape) * epsilon
    return np.argmax(array, axis=1)

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        Q_value = main_model.predict(state, verbose=0)
        return nargmax(Q_value)

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
    state = np.reshape(state, [-1, input_size**2])
    action = int(epsilon_greedy_policy(state, epsilon))
    # try:
    #     action = int(action)
    # except:
    #     xy = action.split(" ")
    #     x = int(xy[0])
    #     y = ord(xy[1])-65
    #     action = env.state.board.coord_to_action(x, y)
    next_state, reward, done, info = env.step(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = target_model.predict(next_states, verbose=0).reshape(-1, 4)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards + (1 - dones) * discount_rate * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, output_size)
    with tf.GradientTape() as tape:
        all_Q_values = main_model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, main_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_model.trainable_variables))

env = gym.make("Omok-v0")

input_size = env.board_size
output_size = env.action_space.n

main_model = keras.models.Sequential([
    keras.layers.Dense(15*15*15, activation="relu", input_shape=[input_size**2]),
    keras.layers.Dense(15*15*3, activation="relu"),
    keras.layers.Dense(output_size),
])

main_model.summary()

target_model = keras.models.clone_model(main_model)
target_model.set_weights(main_model.get_weights())

episodes = 100
batch_size = 32
discount_rate = 0.99
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
loss_fn = keras.losses.Huber()
replay_memory = deque(maxlen=1000000)

for episode in range(episodes):
    state = env.reset()
    epsilon = max(1 - episode / (episodes*2/3.), 0.0)
    rewards = 0
    while True:
        state, reward, done, info = play_one_step(env, state, epsilon)
        if done:
            break
    print(f"\rEpisode: {episode+1} / {episodes}, eps: {epsilon:.3f}, reward: {rewards:.2f}", end="")
    if ((episode+1) >= (episodes*0.1)):
        training_step(batch_size)
    if ((episode+1) % (episodes*0.05) == 0): 
        target_weights = target_model.get_weights()
        online_weights = main_model.get_weights()
        for index in range(len(target_weights)):
           target_weights[index] = 0.99 * target_weights[index] + 0.01 * online_weights[index]
        target_model.set_weights(target_weights)


# for episode in range(episodes):
#     env.reset()
#     for _ in range(20):
#         env.render()
#         # action = input("Type location [EX : (10 H)] : ")
#         action = env.action_space.sample()

#         try:
#             action = int(action)
#         except:
#             xy = action.split(" ")
#             x = int(xy[0])
#             y = ord(xy[1])-65
#             action = env.state.board.coord_to_action(x, y)

#         observation, reward, done, info = env.step(action)
#         if done:
#             print ("Game is Over")
#             break
#         time.sleep(1)