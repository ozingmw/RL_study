import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam


class Actor(Model):
    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        
        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.mu = Dense(action_dim, activation='tanh')
        self.std = Dense(action_dim, activation='softplus')

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        mu = self.mu(x)
        std = self.std(x)

        mu = Lambda(lambda x: x*self.action_bound)(mu)

        return [mu, std]

class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.v = Dense(1, activation='linear')

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        v = self.v(x)

        return v

class PPOagent(object):
    def __init__(self):
        self.DISCOUNT_FACTOR = 0.95
        self.GAE_LAMBDA = 0.9
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.RATIO_CLIPPING = 0.05
        self.EPOCHS = 5

        self.env = gym.make("Pendulum-v1", g=9.80665)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]

        self.actor = Actor(self.action_dim, self.action_bound)
        self.critic = Critic()
        self.actor.build(input_shape=(None, self.state_dim))
        self.critic.build(input_shape=(None, self.state_dim))

        self.actor_optimizer = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_optimizer = Adam(self.CRITIC_LEARNING_RATE)

        self.save_episode_reward = []

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def get_policy_action(self, state):
        mu, std = self.actor(state)
        mu = mu.numpy()[0]
        std = std.numpy()[0]
        std = np.clip(std, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu, std, size=self.action_dim)
        action = np.clip(action, -self.action_bound, self.action_bound)
        return mu, std, action

    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.DISCOUNT_FACTOR * forward_val - v_values[k]
            gae_cumulative = self.DISCOUNT_FACTOR * self.GAE_LAMBDA * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
            return gae, n_step_targets

    # def unpack_batch(self, batch):
    #     unpack = batch[0]
    #     for idx in range(len(batch)-1):
    #         unpack = np.append(unpack, batch[idx+1], axis=0)
    #     return unpack

    def unpack_batch(self, batch_memory):
        batch = [batch_memory[index] for index in range(self.BATCH_SIZE)]
        states, actions, rewards, log_old_policy_pdfs = [
            np.array(
                [experience[field_index][0] for experience in batch]
            ) for field_index in range(4)
        ]
        return states, actions, rewards, log_old_policy_pdfs

    def actor_learn(self, log_old_policy_pdf, states, actions, gaes):
        with tf.GradientTape() as tape:
            mu, std = self.actor(states, training=True)
            log_policy_pdf = self.log_pdf(mu, std, actions)

            ratio = tf.exp(log_policy_pdf - log_old_policy_pdf)
            clipped_ratio = tf.clip_by_value(ratio, 1.0-self.RATIO_CLIPPING, 1.0+self.RATIO_CLIPPING)
            surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
            loss = tf.reduce_mean(surrogate)
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

    def critic_learn(self, states, td_targets):
        with tf.GradientTape() as tape:
            td_hat = self.critic(states, training=True)
            loss = tf.reduce_mean(tf.square(td_hat - td_targets))
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

    def train(self, max_episode_num):
        # batch_state, batch_action, batch_reward, batch_log_old_policy_pdf = [], [], [], []
        batch_memory = deque()

        for episode in range(int(max_episode_num)):
            episode_step, episode_reward, done = 0, 0, False
            state = self.env.reset()

            while not done:
                # self.env.render()
                mu_old, std_old, action = self.get_policy_action(tf.convert_to_tensor([state], dtype=tf.float32))
                var_old = std_old ** 2
                log_old_policy_pdf = -0.5 * (action - mu_old) ** 2 / var_old - 0.5 * np.log(var_old * 2 * np.pi)
                log_old_policy_pdf = np.sum(log_old_policy_pdf)
                
                next_state, reward, done, info = self.env.step(action)
                
                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                reward = np.reshape(reward, [1, 1])
                log_old_policy_pdf = np.reshape(log_old_policy_pdf, [1, 1])

                reward_min = -16.2736
                train_reward = (reward - reward_min/2) / -reward_min * 2

                # batch_state.append(state)
                # batch_action.append(action)
                # batch_reward.append(train_reward)
                # batch_log_old_policy_pdf.append(log_old_policy_pdf)
                batch_memory.append((state, action, train_reward, log_old_policy_pdf))

                if len(batch_memory) < self.BATCH_SIZE:
                    state = next_state
                    episode_reward += reward[0]
                    episode_step += 1
                    continue

                # states = self.unpack_batch(batch_state)
                # actions = self.unpack_batch(batch_action)
                # rewards = self.unpack_batch(batch_reward)
                # log_old_policy_pdfs = self.unpack_batch(batch_log_old_policy_pdf)

                # batch_state, batch_action, batch_reward, batch_log_old_policy_pdf = [], [], [], []

                states, actions, rewards, log_old_policy_pdfs = self.unpack_batch(batch_memory)

                batch_memory.clear()

                next_v_value = self.critic(tf.convert_to_tensor([next_state], dtype=tf.float32))
                v_value = self.critic(tf.convert_to_tensor(states, dtype=tf.float32))
                gaes, y_i = self.gae_target(rewards, v_value.numpy(), next_v_value.numpy(), done)

                for _ in range(self.EPOCHS):
                    self.actor_learn(tf.convert_to_tensor(log_old_policy_pdfs, dtype=tf.float32),
                                     tf.convert_to_tensor(states, dtype=tf.float32),
                                     tf.convert_to_tensor(actions, dtype=tf.float32),
                                     tf.convert_to_tensor(gaes, dtype=tf.float32))
                                     
                    self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                      tf.convert_to_tensor(y_i, dtype=tf.float32))
                
                state = next_state
                episode_reward += reward[0]
                episode_step += 1
                print(f'Episode: {episode+1}, Episode_step: {episode_step}, Reward: {episode_reward[0]:.3f}\r', end="")
        
            print(f'Episode: {episode+1}, Episode_step: {episode_step}, Reward: {episode_reward[0]:.3f}')
            self.save_episode_reward.append(episode_reward)

            if episode % 10 == 0:
                self.actor.save_weights('rl_with_math/model/pendulum_ppo_actor.h5')
                self.critic.save_weights('rl_with_math/model/pendulum_ppo_critic.h5')

        np.savetxt('./rl_with_math/model/pendulum_ppo_episode_reward.txt', self.save_episode_reward, fmt='%.6f')
    
    def plot_result(self):
        plt.plot(self.save_episode_reward)
        plt.show()

max_episode_num = 1000
agent = PPOagent()
agent.train(max_episode_num)
agent.plot_result()