import os
import random
import threading
import time
import gym
import numpy as np
import tensorflow as tf
import keras
import keras.layers
import keras.optimizers

from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.resize_observation import ResizeObservation

def A3C_model(action_size, state_size):
    input = keras.layers.Input(shape=state_size)
    conv1 = keras.layers.Conv2D(32, 8, 4, activation='relu')(input)
    conv2 = keras.layers.Conv2D(64, 4, 2, activation='relu')(conv1)
    conv3 = keras.layers.Conv2D(64, 3, 1, activation='relu')(conv2)
    flatten = keras.layers.Flatten()(conv3)
    shared_fc = keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(flatten)

    policy = keras.layers.Dense(action_size, activation='softmax', kernel_initializer='he_normal')(shared_fc)
    value = keras.layers.Dense(1, activation='linear', kernel_initializer='he_normal')(shared_fc)

    model = keras.Model(inputs=input, outputs=[policy, value])

    return model

class A3CAgent():
    def __init__(self, env_name, action_size):
        self.env_name = env_name
        self.action_size = action_size
        self.state_size = (84, 84, 4)
        
        self.discount_factor = 0.99
        self.lr = 1e-4
        self.threads = 16
        
        self.global_model = A3C_model(self.action_size, self.state_size)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr, clipnorm=5.)
        # set actor/critic optimizer

        # self.model_path = os.path.join(os.getcwd(), 'save_model')

    def train(self):
        runners = [Runner(self.env_name, self.action_size, self.state_size,
                          self.global_model, self.discount_factor,
                          self.optimizer) 
                   for _ in range(self.threads)]
        for i, runner in enumerate(runners):
            print(f'Start worker #{i+1}')
            runner.start()

        # while True:
        #     self.global_model.save_weights(self.model_path)
        #     time.sleep(10*60)

class Runner(threading.Thread):
    global_episode = 0

    def __init__(self, env_name, action_size, state_size, global_model,
                 discount_factor, optimizer):
        threading.Thread.__init__(self)
        self._make_env(env_name)
        self.action_size = action_size
        self.state_size = state_size
        self.global_model = global_model
        self.discount_factor = discount_factor
        self.optimizer = optimizer

        self.states, self.actions, self.rewards = [], [], []

        self.local_model = A3C_model(action_size, state_size)
        self.agent_waiting = 20
        self.t = 0
        self.t_max = 20

    def _make_env(self, env_name):
        self.env = GrayScaleObservation(gym.make(env_name), keep_dim=True)
        self.env = ResizeObservation(self.env, 84)

    def get_action(self, history):
        history = np.float32(history / 255.)
        policy = self.local_model(history)[0][0]
        policy = np.array(policy)
        policy = np.round(policy, 6)
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        return action

    def train_model(self, done):
        global_params = self.global_model.trainable_variables
        local_params = self.local_model.trainable_variables

        with tf.GradientTape() as tape:
            total_loss = self.compute_loss(done)
        grads = tape.gradient(total_loss, local_params)
        self.optimizer.apply_gradients(zip(grads, global_params))
        self.local_model.set_weights(self.global_model.get_weights())
        self.states, self.actions, self.rewards = [], [], []

    def compute_loss(self, done):
        discounted_prediction = self.discounted_prediction(self.rewards, done)
        discounted_prediction = tf.convert_to_tensor(discounted_prediction[:, None], dtype=tf.float32)

        states = np.zeros((len(self.states), 84, 84, 4))

        for i in range(len(self.states)):
            states[i] = self.states[i]
        states = np.float32(states/255.)

        policy, values = self.local_model(states)

        advantages = discounted_prediction - values
        critic_loss = 0.5 * tf.reduce_sum(tf.square(advantages))

        action = tf.convert_to_tensor(self.actions, dtype=tf.float32)
        policy_prob = tf.nn.softmax(policy)
        action_prob = tf.reduce_sum(action * policy_prob, axis=1, keepdims=True)
        cross_entropy = -tf.math.log(action_prob + 1e-10)
        actor_loss = tf.reduce_sum(cross_entropy * tf.stop_gradient(advantages))

        entropy = tf.reduce_sum(policy_prob * tf.math.log(policy_prob + 1e-10), axis=1)
        entropy = tf.reduce_sum(entropy)
        actor_loss += 0.01 * entropy

        total_loss = 0.5 * critic_loss + actor_loss

        return total_loss

    def discounted_prediction(self, rewards, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0

        if not done:
            last_state = np.float32(self.states[-1]/255.)
            running_add = self.local_model(last_state)[-1][0]

        for t in reversed(range(0, len(rewards))):
            running_add *= self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add

        return discounted_prediction

    def append_sample(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    def run(self):
        global episode, score_avg, score_max
        
        while episode < max_episode:
            obs = self.env.reset()
            score, life = 0, 5
            dead = False

            for _ in range(random.randint(1, self.agent_waiting)):
                obs, _, _, done = self.env.step(1)

            history = np.stack([obs, obs, obs, obs], axis=2)
            history = np.reshape([history], (1, 84, 84, 4))

            while True:
                # self.env.render(mode="rgb_array")
                self.t += 1
                
                action = self.get_action(history)
                # action space
                # ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
                # 0, 1, 2 -> 0, 2, 3으로 변경
                real_action = action + 1 if action != 0 else action
                if dead:
                    action, real_action, dead = 0, 1, False
                
                obs, reward, done, info = self.env.step(real_action)

                next_state = obs
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                if life > info['lives']:
                    dead = True
                    life = info['lives']

                score += reward
                reward = np.clip(reward, -1., 1.)

                self.append_sample(history, action, reward)

                if dead:
                    history = np.stack((next_state, next_state, next_state, next_state))
                    history = np.reshape([history], (1, 84, 84, 4))
                else:
                    history = next_history

                if self.t >= self.t_max or done:
                    self.train_model(done)
                    self.t = 0

                if done:
                    episode += 1
                    score_max = score if score > score_max else score_max
                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score

                    print(f"episode: {episode:5d} | score: {score:4.1f} | score max: {score_max:4.1f} | score avg: {score_avg:.3f}")

                    break

global episode, score_max, score_avg
episode, score_max, score_avg = 0, 0, 0
max_episode = 10000000

env_name = 'BreakoutDeterministic-v4'
global_agent = A3CAgent(env_name, action_size=3)
global_agent.train()