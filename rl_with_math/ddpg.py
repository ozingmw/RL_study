import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from keras.layers import Dense, Lambda, Concatenate, Input
from keras.models import Model
from keras.optimizers import Adam


class Actor(Model):
    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        
        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.action = Dense(action_dim, activation='tanh')

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        action = self.action(x)
        action = Lambda(lambda x: x*self.action_bound)(action)

        return action

class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()

        self.x1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.q = Dense(1, activation='linear')

    def call(self, state_action):
        state, action = state_action[0], state_action[1]
        
        x = self.x1(state)
        x_a = Concatenate(axis=-1)([x, action])
        v = self.h2(x_a)
        v = self.h3(v)
        v = self.q(v)

        return v

class DDPGagent(object):
    def __init__(self):
        self.DISCOUNT_FACTOR = 0.95
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 20000
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.TAU = 0.001

        self.env = gym.make("Pendulum-v1", g=9.80665)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]

        self.actor = Actor(self.action_dim, self.action_bound)
        self.target_actor = Actor(self.action_dim, self.action_bound)
        self.critic = Critic()
        self.target_critic = Critic()

        self.actor.build(input_shape=(None, self.state_dim))
        self.target_actor.build(input_shape=(None, self.state_dim))
        
        state_in = Input((self.state_dim,))
        action_in = Input((self.action_dim,))
        self.critic([state_in, action_in])
        self.target_critic([state_in, action_in])

        self.actor_optimizer = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_optimizer = Adam(self.CRITIC_LEARNING_RATE)

        self.buffer = deque(maxlen=self.BUFFER_SIZE)

        self.save_episode_reward = []

    def update_target_network(self, TAU):
        weights = self.actor.get_weights()
        target_weights = self.target_actor.get_weights()
        for index in range(len(weights)):
            target_weights[index] = TAU * weights[index] + (1-TAU) * target_weights[index]
        self.target_actor.set_weights(target_weights)

        weights = self.critic.get_weights()
        target_weights = self.target_critic.get_weights()
        for index in range(len(weights)):
            target_weights[index] = TAU * weights[index] + (1-TAU) * target_weights[index]
        self.target_critic.set_weights(target_weights)

    def ou_noise(self, x, rho=0, mu=0, dt=1e-1, sigma=0.2, loc=0, scale=1, dim=1):
        return x + rho*(mu - x)*dt + sigma*np.sqrt(dt)*np.random.normal(loc=loc, scale=scale, size=dim)

    def td_target(self, rewards, v_values, dones):
        td = np.asarray(v_values)
        for index in range(v_values.shape[0]):
            td[index] = rewards[index] + (1-dones[index]) * self.DISCOUNT_FACTOR * v_values[index]
        return td
        
    def unpack_batch(self, replay_memory):
        indices = np.random.randint(len(replay_memory), size=self.BATCH_SIZE)
        batch = [replay_memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array(
                [experience[field_index] for experience in batch]
            ) for field_index in range(5)
        ]
        return states, actions, rewards, next_states, dones

    def actor_learn(self, states):
        with tf.GradientTape() as tape:
            actions = self.actor(states, training=True)
            critic = self.critic([states, actions])
            loss = -tf.reduce_mean(critic)
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

    def critic_learn(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            v = self.critic([states, actions], training=True)
            loss = tf.reduce_mean(tf.square(v - td_targets))
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

    def train(self, max_episode_num):
        self.update_target_network(1.0)

        for episode in range(int(max_episode_num)):
            pre_noise = np.zeros(self.action_dim)
            episode_step, episode_reward, done = 0, 0, False
            state = self.env.reset()

            while not done:
                # self.env.render()
                action = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
                action = action.numpy()[0]
                noise = self.ou_noise(pre_noise, dim=self.action_dim)
                action = np.clip(action + noise, -self.action_bound, self.action_bound)

                next_state, reward, done, info = self.env.step(action)
                
                reward_min = -16.2736044
                train_reward = (reward - reward_min/2) / -reward_min * 2

                self.buffer.append((state, action, train_reward, next_state, done))

                if len(self.buffer) > 1000:
                    states, actions, rewards, next_states, dones = self.unpack_batch(self.buffer)
                    target_qs = self.target_critic([
                        tf.convert_to_tensor(next_states, dtype=tf.float32),
                        self.target_actor(tf.convert_to_tensor(next_states, dtype=tf.float32))
                    ])

                    y_i = self.td_target(rewards, target_qs.numpy(), dones)

                    self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                      tf.convert_to_tensor(actions, dtype=tf.float32),
                                      tf.convert_to_tensor(y_i, dtype=tf.float32))
                    self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32))
                    self.update_target_network(self.TAU)

                pre_noise = noise
                state = next_state
                episode_reward += reward
                episode_step += 1
                print(f'Episode: {episode+1}, Episode_step: {episode_step}, Reward: {episode_reward:.3f}\r', end="")
        
            print(f'Episode: {episode+1}, Episode_step: {episode_step}, Reward: {episode_reward:.3f}')
            self.save_episode_reward.append(episode_reward)

            if episode % 10 == 0:
                self.actor.save_weights('rl_with_math/model/pendulum_ddpg_actor.h5')
                self.critic.save_weights('rl_with_math/model/pendulum_ddpg_critic.h5')

        np.savetxt('./rl_with_math/model/pendulum_ppo_episode_reward.txt', self.save_episode_reward, fmt='%.6f')
    
    def plot_result(self):
        plt.plot(self.save_episode_reward)
        plt.show()

max_episode_num = 200
agent = DDPGagent()
agent.train(max_episode_num)
agent.plot_result()