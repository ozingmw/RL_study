import gym
import numpy as np
import tensorflow as tf
import keras

import matplotlib.pyplot as plt
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Lambda

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

class A2CAgent(object):
    def __init__(self, env):
        self.DISCOUNT_FACTOR = 0.95
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001

        self.env = env

        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.action_bound = env.action_space.high[0]
        self.std_bound = [0.01, 1.0]
        
        self.actor = Actor(self.action_dim, self.action_bound)
        self.critic = Critic()
        self.actor.build(input_shape=(None, self.observation_dim))
        self.critic.build(input_shape=(None, self.observation_dim))
        
        # self.actor.summary()
        # self.critic.summary()

        self.actor_optimizer = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_optimizer = Adam(self.CRITIC_LEARNING_RATE)
        
        self.save_episode_reward = []

    def train(self, max_episode_num):
        for episode in range(max_episode_num):
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = [], [], [], [], []
            episode_step, episode_reward, done = 0, 0, False
            state = self.env.reset()

            while not done:
                # self.env.render()
                action = self.get_action(tf.convert_to_tensor([state], dtype=tf.float32))
                action = np.clip(action, -self.action_bound, self.action_bound)
                
                next_state, reward, done, info = self.env.step(action)
                
                state = np.reshape(state, [1, self.observation_dim])
                action = np.reshape(action, [1, self.action_dim])
                reward = np.reshape(reward, [1, 1])
                next_state = np.reshape(next_state, [1, self.observation_dim])
                done = np.reshape(done, [1, 1])

                # reward [-16.2736, 0] -> [-1, 1]
                reward_min = -16.2736
                train_reward = (reward - reward_min/2) / -reward_min * 2

                batch_state.append(state)
                batch_action.append(action)
                batch_reward.append(train_reward)
                batch_next_state.append(next_state)
                batch_done.append(done)

                if len(batch_state) < self.BATCH_SIZE:
                    state = next_state[0]
                    episode_reward += reward[0]
                    episode_step += 1
                    continue
            
                states = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                train_rewards = self.unpack_batch(batch_reward)
                next_states = self.unpack_batch(batch_next_state)
                dones = self.unpack_batch(batch_done)

                batch_state, batch_action, batch_reward, batch_next_state, batch_done = [], [], [], [], []
                
                next_v_values = self.critic(tf.convert_to_tensor(next_states, dtype=tf.float32))
                td_targets = self.td_target(train_rewards, next_v_values.numpy(), dones)

                self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32), tf.convert_to_tensor(td_targets, dtype=tf.float32))

                v_values = self.critic(tf.convert_to_tensor(states, dtype=tf.float32))
                next_v_values = self.critic(tf.convert_to_tensor(next_states, dtype=tf.float32))
                advantages = train_rewards + self.DISCOUNT_FACTOR * next_v_values - v_values

                self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                tf.convert_to_tensor(actions, dtype=tf.float32),
                                tf.convert_to_tensor(advantages, dtype=tf.float32))

                state = next_state[0]
                episode_reward += reward[0]
                episode_step += 1
        
            print(f'Episode: {episode+1}, Episode_step: {episode_step}, Reward: {episode_reward[0]:.3f}')

            self.save_episode_reward.append(episode_reward)

            if episode % 10 == 0:
                self.actor.save_weights('rl_with_math/model/pendulum_actor.h5')
                self.critic.save_weights('rl_with_math/model/pendulum_critic.h5')

        np.savetxt('./rl_with_math/model.pendulum_episode_reward.txt', self.save_episode_reward)
        print(self.save_episode_reward)

    def get_action(self, state):
        mu, std = self.actor(state)
        mu = mu.numpy()[0]
        std = std.numpy()[0]
        std = np.clip(std, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu, std, size=self.action_dim)
        return action

    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch)-1):
            unpack = np.append(unpack, batch[idx+1], axis=0)
        return unpack

    def td_target(self, rewards, next_v_values, dones):
        y_i = np.zeros(next_v_values.shape)
        for i in range(next_v_values.shape[0]):
            y_i[i] = rewards[i] + (1 - dones[i]) * self.DISCOUNT_FACTOR * next_v_values[i]
        return y_i

    def critic_learn(self, states, td_targets):
        with tf.GradientTape() as tape:
            td_hat = self.critic(states, training=True)
            loss = tf.reduce_mean(tf.square(td_targets - td_hat))
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

    def actor_learn(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            mu, std = self.actor(states, training=True)
            log_policy_pdf = self.log_pdf(mu, std, actions)
            loss_policy = log_policy_pdf * advantages
            loss = tf.reduce_mean(-loss_policy)
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        
    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def plot_result(self):
        plt.plot(self.save_episode_reward)
        plt.show()

    def load_weights(self):
        self.actor.load_weights('rl_with_math/model/pendulum_actor.h5')
        self.critic.load_weights('rl_with_math/model/pendulum_critic.h5')
    
    def load_weights(self, path):
        self.actor.load_weights(path + '/pendulum_actor.h5')
        self.critic.load_weights(path + '/pendulum_critic.h5')

max_episode_num = 1000
env = gym.make('Pendulum-v1', g=9.80665)
a2c = A2CAgent(env)
a2c.train(max_episode_num)
a2c.plot_result()