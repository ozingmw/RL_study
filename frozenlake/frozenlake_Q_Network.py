from collections import deque
import gym
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(output_size)
    else:
        Q_value = main_model.predict(np.array(state).reshape(-1, 1), verbose=0)
        return np.argmax(Q_value)
        # return np.random.choice(output_size, 1, p=Q_value[0])[0]

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
    # if next_state == state:
    #     reward -= 0.007
    # if done and (reward == 0):
    #     reward -= 0.01
    # elif not done:
    #     reward += 0.002
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = target_model.predict(next_states.reshape(-1, 1), verbose=0).reshape(-1, 4)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards + (1 - dones) * discount_rate * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, output_size)
    with tf.GradientTape() as tape:
        all_Q_values = tf.reshape(main_model(states.reshape(-1, 1)), [-1, output_size])
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
        # loss_list.append(loss)
    # print(f", loss: {loss:.4f}", end="")
    grads = tape.gradient(loss, main_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_model.trainable_variables))

def bot_render(main_model, repeat=1):
    total_reward = []
    for _ in range(repeat):    
        state = env.reset()
        print(f"repeat: {_+1}")
        middle_reward = 0
        while True:
            env.render()
            action = np.argmax(tf.reshape(main_model.predict(np.array(state).reshape(-1, 1), verbose=0), [-1, output_size]))
            state, reward, done, info = env.step(action)
            middle_reward += reward
            if done:
                break
        total_reward.append(middle_reward)
    print(f"{sum(total_reward)/repeat:.3f}")


env = gym.make("FrozenLake-v1")

input_size = env.observation_space.n
output_size = env.action_space.n

def make_model():
    model = keras.models.Sequential([
        keras.layers.Dense(10, activation="relu", input_shape=[None, 1]),
        keras.layers.Dense(output_size),
    ])
    return model

main_model = make_model()
target_model = make_model()

episodes = 1000
batch_size = 50
discount_rate = 0.95
optimizer = keras.optimizers.RMSprop(learning_rate=1e-2)
loss_fn = keras.losses.Huber()
replay_memory = deque(maxlen=2000)

rewards_list = []
loss_list = []

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
    if (episode % (episodes*0.05) == 0):
        for _ in range(int(episodes*0.05)):
            training_step(batch_size)
        target_model.set_weights(main_model.get_weights())    
        print(main_model(np.array([range(16)]))[0])

# fig, axes = plt.subplots(1, 2)
sns.lineplot(range(1, len(rewards_list)+1), rewards_list, label="reward", color="red")
# sns.lineplot(range(1, len(move_list)+1), move_list, label="move", color="blue", ax=axes[1])
sns.set(style="darkgrid")
plt.show()

bot_render(main_model, 5)