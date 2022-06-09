from collections import deque
from datetime import datetime
import gym
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns

#one hot vector giveup
def to_one_hot(states):
    t1 = tf.one_hot(states[0]-1, input_size[0])
    t2 = tf.one_hot(states[1]-1, input_size[1])
    t3 = tf.constant(tf.one_hot(int(states[2] == False), input_size[2]))
    return tf.concat([t1, t2, t3], axis=1)

def nargmax(array):
    epsilon = 1e-6
    array += np.random.rand(*array.shape) * epsilon
    return np.argmax(array, axis=1)
    
def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(output_size)
    else:
        Q_value = main_model.predict(to_one_hot(state))
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
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = target_model.predict(to_one_hot(next_states), verbose=0).reshape(-1, 4)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards + (1 - dones) * discount_rate * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, output_size)
    with tf.GradientTape() as tape:
        all_Q_values = tf.reshape(main_model(to_one_hot(states)), [-1, output_size])
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, main_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_model.trainable_variables))

def bot_render(model, repeat=1):
    total_reward = []
    for _ in range(repeat):    
        state = env.reset()
        print(f"repeat: {_+1}")
        middle_reward = 0
        while True:
            env.render()
            action = nargmax(main_model.predict(tf.reshape(to_one_hot(state), [-1, input_size]), verbose=0))
            state, reward, done, info = env.step(int(action))
            middle_reward += reward
            if done:
                break
        total_reward.append(middle_reward)
        print(f"{middle_reward}")
    print(f"{sum(total_reward)/repeat:.3f}")


env = gym.make("Blackjack-v1", natural=True)

input_size = [x.n for x in env.observation_space]
output_size = env.action_space.n

main_model = keras.models.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=input_size),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(output_size),
])

target_model = keras.models.clone_model(main_model)
target_model.set_weights(main_model.get_weights())

episodes = 2000
batch_size = 32
discount_rate = 0.99
optimizer = keras.optimizers.Adam(learning_rate=0.025, clipnorm=1.0)
loss_fn = keras.losses.Huber()
replay_memory = deque(maxlen=100000)

rewards_list = []
# success = 0

for episode in range(episodes):
    state = env.reset()
    epsilon = max(1 - episode / (episodes*2/3.), 0.01)
    rewards = 0
    while True:
        state, reward, done, info = play_one_step(env, state, epsilon)
        rewards += reward
        if done:
            break
    print(f"\rEpisode: {episode+1} / {episodes}, eps: {epsilon:.3f}, reward: {rewards:.2f}", end="")
    rewards_list.append(rewards)
    if ((episode+1) >= (episodes*0.1)):
        training_step(batch_size)
    if ((episode+1) % (episodes*0.05) == 0): 
        target_weights = target_model.get_weights()
        online_weights = main_model.get_weights()
        for index in range(len(target_weights)):
           target_weights[index] = 0.99 * target_weights[index] + 0.01 * online_weights[index]
        target_model.set_weights(target_weights)
        # target_model.set_weights(main_model.get_weights())  
        print(main_model(to_one_hot(np.array([range(16)])))[0])

now = datetime.now().strftime("%y%m%d_%H%M")
main_model.save(f"./model/blackjack/blackjack_{now}.h5")

# fig, axes = plt.subplots(1, 2)
sns.lineplot(range(1, len(rewards_list)+1), rewards_list, label="reward", color="red")
# sns.lineplot(range(1, len(move_list)+1), move_list, label="move", color="blue", ax=axes[1])
sns.set(style="darkgrid")
plt.title(f"score : {(sum(rewards_list) / len(rewards_list) * 100):.3f}%")
plt.savefig(f"./img_log/blackjack/{now}.png", dpi=300)
plt.show()

bot_render(main_model, 5)