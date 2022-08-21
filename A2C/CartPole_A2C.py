from datetime import datetime
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

count = 3
min_episode_count = 10000

for _ in range(count):
    print(f"Env: {_+1}")
    gamma = 0.99
    max_steps_per_episode = 10000
    eps = np.finfo(np.float32).eps

    env = gym.make("CartPole-v1", edit=True)

    num_inputs = 4
    num_actions = 2
    num_hidden = 128

    inputs = keras.layers.Input(shape=(num_inputs,))
    dense = keras.layers.Dense(num_hidden, activation="relu")(inputs)
    action = keras.layers.Dense(num_actions, activation="softmax")(dense)
    critic = keras.layers.Dense(1)(dense)
    model = keras.Model(inputs=inputs, outputs=[action, critic])

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    huber_loss = keras.losses.Huber()
    action_probs_history = []
    critic_value_history = []
    rewards_history = []
    running_reward = 0
    episode_count = 0

    while True:
        state = env.reset()
        episode_reward = 0
        with tf.GradientTape() as tape:
            for timestep in range(1, max_steps_per_episode+1):
                # env.render()
                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)

                action_probs, crititc_value = model(state)
                critic_value_history.append(crititc_value[0, 0])

                action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                action_probs_history.append(tf.math.log(action_probs[0, action]))

                state, reward, done, info = env.step(action)
                episode_reward += reward
                rewards_history.append(episode_reward)

                if done:
                    break

            running_reward = 0.05 * episode_reward + (1- 0.05) * running_reward

            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            history = zip(action_probs_history, critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                diff = ret - value
                actor_losses.append(-log_prob * diff)
                critic_losses.append(huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))
            
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()

        episode_count+= 1
        if episode_count % 10 == 0:
            print(f"running reward: {running_reward:.2f} at episode {episode_count}")

        if running_reward >= 475:
            print(f"Solved at episode {episode_count}! ")
            if min_episode_count >= episode_count:
                best_model = keras.models.clone_model(model)
                min_episode_count = episode_count
            break


now = datetime.now().strftime("%y%m%d_%H%M")
best_model.save(f"./model/CartPole/CartPole_A2C_{now}.h5")
print("Model Saved!")

state = env.reset()
model = keras.models.load_model(f"./model/CartPole/CartPole_A2C_{now}", compile=False)
while True:    
    episode_reward = 0
    env.render()
    state = tf.convert_to_tensor(state)
    state = tf.expand_dims(state, 0)

    action_probs, _ = model(state)
    action = np.random.choice(num_actions, p=np.squeeze(action_probs))

    state, reward, done, info = env.step(action)

    if done:
        break