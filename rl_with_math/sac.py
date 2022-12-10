import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Lambda, Concatenate, Input
from keras.models import Model
from keras.optimizers import Adam
import tensorflow_probability as tfp


class Actor(Model):
    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.std_bound = [1e-2, 1.0]

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
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])

        return mu, std

    def sample_normal(self, mu, std):
        normal_prob = tfp.distributions.Normal(mu, std)
        action = normal_prob.sample()
        action = tf.clip_by_value(action, -self.action_bound, self.action_bound)
        log_pdf = normal_prob.log_prob(action)
        log_pdf = tf.reduce_sum(log_pdf, 1, keepdims=True)
        return action, log_pdf

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

class SACagent(object):
    def __init__(self):
        self.DISCOUNT_FACTOR = 0.95
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 20000
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.TAU = 0.001
        self.ALPHA = 0.5

        self.env = gym.make("Pendulum-v1", g=9.80665)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]

        self.actor = Actor(self.action_dim, self.action_bound)
        self.actor.build(input_shape=(None, self.state_dim))
        
        self.critic_1 = Critic()
        self.critic_2 = Critic()
        self.target_critic_1 = Critic()
        self.target_critic_2 = Critic()
        
        state_in = Input((self.state_dim,))
        action_in = Input((self.action_dim,))
        self.critic_1([state_in, action_in])
        self.critic_2([state_in, action_in])
        self.target_critic_1([state_in, action_in])
        self.target_critic_2([state_in, action_in])


        self.actor_optimizer = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_1_optimizer = Adam(self.CRITIC_LEARNING_RATE)
        self.critic_2_optimizer = Adam(self.CRITIC_LEARNING_RATE)

        self.buffer = deque(maxlen=self.BUFFER_SIZE)

        self.save_episode_reward = []

    def get_action(self, state):
        mu, std = self.actor(state)
        action, _ = self.actor.sample_normal(mu, std)
        return action.numpy()[0]

    def update_target_network(self, TAU):
        weights_1 = self.critic_1.get_weights()
        weights_2 = self.critic_2.get_weights()
        target_weights_1 = self.target_critic_1.get_weights()
        target_weights_2 = self.target_critic_2.get_weights()
        for index in range(len(weights_1)):
            target_weights_1[index] = TAU * weights_1[index] + (1-TAU) * target_weights_1[index]
            target_weights_2[index] = TAU * weights_2[index] + (1-TAU) * target_weights_2[index]
        self.target_critic_1.set_weights(target_weights_1)
        self.target_critic_2.set_weights(target_weights_2)

    def unpack_batch(self, replay_memory):
        indices = np.random.randint(len(replay_memory), size=self.BATCH_SIZE)
        batch = [replay_memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array(
                [experience[field_index] for experience in batch]
            ) for field_index in range(5)
        ]
        return states, actions, rewards, next_states, dones

    def q_target(self, rewards, q_values, dones):
        y_k = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            y_k[i] = rewards[i] + (1-dones[i]) * self.DISCOUNT_FACTOR * q_values[i]
        return y_k

    def actor_learn(self, states):
        with tf.GradientTape() as tape:
            mu, std = self.actor(states, training=True)
            actions, log_pdfs = self.actor.sample_normal(mu, std)
            log_pdfs = tf.squeeze(log_pdfs, 1)
            soft_q_1 = self.critic_1([states, actions])
            soft_q_2 = self.critic_2([states, actions])
            soft_q = tf.math.minimum(soft_q_1, soft_q_2)
            loss = tf.reduce_mean(self.ALPHA * log_pdfs - soft_q)
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

    def critic_learn(self, states, actions, q_targets):
        with tf.GradientTape() as tape:
            q_1 = self.critic_1([states, actions], training=True)
            loss_1 = tf.reduce_mean(tf.square(q_1 - q_targets))
        grads_1 = tape.gradient(loss_1, self.critic_1.trainable_variables)
        self.critic_1_optimizer.apply_gradients(zip(grads_1, self.critic_1.trainable_variables))
        
        with tf.GradientTape() as tape:
            q_2 = self.critic_2([states, actions], training=True)
            loss_2 = tf.reduce_mean(tf.square(q_2 - q_targets))
        grads_2 = tape.gradient(loss_2, self.critic_2.trainable_variables)
        self.critic_2_optimizer.apply_gradients(zip(grads_2, self.critic_2.trainable_variables))

    def train(self, max_episode_num):
        self.update_target_network(1.0)

        for episode in range(int(max_episode_num)):
            episode_step, episode_reward, done = 0, 0, False
            state = self.env.reset()

            while not done:
                # self.env.render()
                action = self.get_action(tf.convert_to_tensor([state], dtype=tf.float32))
                action = np.clip(action, -self.action_bound, self.action_bound)

                next_state, reward, done, info = self.env.step(action)
                
                reward_min = -16.2736044
                train_reward = (reward - reward_min/2) / -reward_min * 2

                self.buffer.append((state, action, train_reward, next_state, done))

                if len(self.buffer) > 1000:
                    states, actions, rewards, next_states, dones = self.unpack_batch(self.buffer)
                    next_mu, next_std = self.actor(tf.convert_to_tensor(next_states, dtype=tf.float32))
                    next_actions, next_log_pdf = self.actor.sample_normal(next_mu, next_std)

                    target_qs_1 = self.target_critic_1([next_states, next_actions])
                    target_qs_2 = self.target_critic_2([next_states, next_actions])
                    target_qs = tf.math.minimum(target_qs_1, target_qs_2)

                    target_qi = target_qs - self.ALPHA * next_log_pdf

                    y_i = self.q_target(rewards, target_qi.numpy(), dones)

                    self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                      tf.convert_to_tensor(actions, dtype=tf.float32),
                                      tf.convert_to_tensor(y_i, dtype=tf.float32))
                    self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32))
                    self.update_target_network(self.TAU)

                state = next_state
                episode_reward += reward
                episode_step += 1
                print(f'Episode: {episode+1}, Episode_step: {episode_step}, Reward: {episode_reward:.3f}\r', end="")
        
            print(f'Episode: {episode+1}, Episode_step: {episode_step}, Reward: {episode_reward:.3f}')
            self.save_episode_reward.append(episode_reward)

            if episode % 10 == 0:
                self.actor.save_weights('rl_with_math/model/pendulum_sac_actor.h5')
                self.critic_1.save_weights('rl_with_math/model/pendulum_sac_critic_1.h5')
                self.critic_2.save_weights('rl_with_math/model/pendulum_sac_critic_2.h5')

        np.savetxt('./rl_with_math/model/pendulum_sac_episode_reward.txt', self.save_episode_reward, fmt='%.6f')
    
    def plot_result(self):
        plt.plot(self.save_episode_reward)
        plt.show()

max_episode_num = 200
agent = SACagent()
agent.train(max_episode_num)
agent.plot_result()