import gym
import numpy as np
import tensorflow as tf
import keras
import threading
import multiprocessing
import time
import matplotlib.pyplot as plt

from keras.layers import Dense, Lambda
from keras.optimizers import Adam
from keras.models import Model


global_episode_count = 0
global_step = 0
global_episode_reward = []


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

class A3Cagent(object):
    def __init__(self):
        self.WORKERS_NUM = multiprocessing.cpu_count()
        
        self.env = gym.make('Pendulum-v1', g=9.81)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]

        self.global_actor = Actor(self.action_dim, self.action_bound)
        self.global_critic = Critic()
        self.global_actor.build(input_shape=(None, self.state_dim))
        self.global_critic.build(input_shape=(None, self.state_dim))

    def load_weights(self, path):
        self.global_actor.load_weights(path + 'a3c_pendulum_actor.h5')
        self.global_critic.load_weights(path + 'a3c_pendulum_critic.h5')

    def train(self, max_episode_num):
        workers = []
        runners = [A3Cworker(self.env, self.state_dim,
                             self.action_dim, self.action_bound,
                             self.global_actor, self.global_critic) 
                             for _ in range(self.WORKERS_NUM)]
        for runner in runners:
            runner.start()
        
        while True:
            self.global_actor.save_weights(path)
            self.global_critic.save_weights(path)
            time.sleep(10*60)
    
    def plot_result(self):
        plt.plot(global_episode_reward)
        plt.show()

class A3Cworker(threading.Thread):
    def __init__(self, env, state_dim, action_dim, action_bound, global_actor, global_critic, max_episode_num):
        threading.Thread.__init__(self)

        self.DISCOUNT_FACTOR = 0.95
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.t_MAX = 4
        
        self.max_episode_num = max_episode_num

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = [1e-3, 1.0]
        self.global_actor = global_actor
        self.global_critic = global_critic

        self.worker_actor = Actor(self.action_dim, self.action_bound)
        self.worker_critic = Critic()
        self.worker_actor.build(input_shape=(None, self.state_dim))
        self.worker_critic.build(input_shape=(None, self.state_dim))

        self.actor_optimizer = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_optimizer = Adam(self.CRITIC_LEARNING_RATE)

        self.worker_actor.set_weights(self.global_actor.get_weights())
        self.worker_critic.set_weights(self.global_critic.get_weights())

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action-mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def get_action(self, state):
        mu, std = self.worker_actor(state)
        mu = mu.numpy()[0]
        std = std.numpy()[0]
        std = np.clip(std, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu, std, size=self.action_dim)
        return action

    def actor_loss(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            mu, std = self.worker_actor(states, training=True)
            log_policy_pdf = self.log_pdf(mu, std, actions)
            
            loss_policy = log_policy_pdf * advantages
            loss = tf.reduce_sum(-loss_policy)
        grads = tape.gradient(loss, self.worker_actor.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 20)
        self.actor_optimizer.apply_gradients(zip(grads, self.global_actor.trainable_variables))

    def critic_loss(self, states, n_step_tf_targets):
        with tf.GradientTape() as tape:
            td_hat = self.worker_critic(states, training=True)
            loss = tf.reduce_mean(tf.square(n_step_tf_targets))
        grads = tape.gradient(loss, self.worker_critic.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 20)
        self.critic.optimizer.apply_gradients(zip(grads, self.global_critic.trainable_variables))

    def n_step_td_target(self, rewards, next_v_value, done):
        