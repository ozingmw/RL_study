from collections import deque
from datetime import datetime
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

def nargmax_by_legal_action(array, legal_action):
    epsilon = 1e-6
    array += np.random.rand(*array.shape) * epsilon
    mask = np.zeros_like(array)
    for temp in legal_action:
        mask[0][temp] = 1
    legal_array = [float("-inf") if mask[0][array_index] == 0 else array[0][array_index] for array_index in range(len(array[0]))]
    legal_array = np.reshape(legal_array, [1, -1])
    return np.argmax(legal_array, axis=1) 

def epsilon_greedy_policy(env, state, epsilon=0):
    legal_action = env.state.board.get_legal_action()
    if np.random.rand() < epsilon:
        return np.random.choice(legal_action, 1)
    else:
        Q_value = main_model.predict(state, verbose=0)
        return nargmax_by_legal_action(Q_value, legal_action)

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
    state = np.reshape(state, [-1, input_size**2])  # shape : [1, 15*15]
    action = int(epsilon_greedy_policy(env, state, epsilon))
    # try:
    #     action = int(action)
    # except:
    #     xy = action.split(" ")
    #     x = int(xy[0])
    #     y = ord(xy[1])-65
    #     action = env.state.board.coord_to_action(x, y)
    next_state, reward, done, info = env.step(action)
    next_state = np.reshape(next_state, [-1, input_size**2])    # shape : [1, 15*15]
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = target_model.predict(np.reshape(next_states, [batch_size, -1]), verbose=0)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards + (1 - dones) * discount_rate * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, output_size)
    with tf.GradientTape() as tape:
        all_Q_values = main_model(np.reshape(states, [batch_size, -1]))
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

target_model = keras.models.clone_model(main_model)
target_model.set_weights(main_model.get_weights())

now = datetime.now().strftime("%y%m%d_%H%M")
path = f"./model/omok/omok_{now}.h5"

episodes = 20000
batch_size = 32
discount_rate = 0.99
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
loss_fn = keras.losses.Huber()
replay_memory = deque(maxlen=5000000)

total_reward = 0
prev_average_reward = float("-inf")

for episode in range(episodes):
    state = env.reset()
    epsilon = max(1 - episode / (episodes*2/3.), 0.001)
    rewards = 0
    # env.render()
    while True:
        state, reward, done, info = play_one_step(env, state, epsilon)
        rewards += reward
        total_reward += reward
        # env.render()
        if done:
            break
        # time.sleep(1)
    now_average_reward = total_reward/(episode+1)
    print(f"\rEpisode: {episode+1} / {episodes}, eps: {epsilon:.3f}, reward: {rewards:.2f}, average_reward: {now_average_reward:.4f}", end="")
    if ((episode+1) >= (episodes*0.1)):
        training_step(batch_size)
    if ((episode+1) % (episodes*0.05) == 0): 
        target_weights = target_model.get_weights()
        online_weights = main_model.get_weights()
        for index in range(len(target_weights)):
           target_weights[index] = 0.99 * target_weights[index] + 0.01 * online_weights[index]
        target_model.set_weights(target_weights)
        env.render()
    if prev_average_reward < now_average_reward:
        main_model.save(path)
        prev_average_reward = now_average_reward

env.reset()
env.ai_opponent(path)
env.render()
while True:
    action = input("Type location [EX : (10 H)] : ")
    
    try:
        action = int(action)
    except:
        xy = action.split(" ")
        x = int(xy[0])
        y = ord(xy[1])-65
        action = env.state.board.coord_to_action(x, y)

    observation, reward, done, info = env.step(action)
    env.render()
    if done:
        print ("Game is Over")
        break