import sys
import gym
import pylab
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

import tensorflow as tf
import tensorflow_probability as tfp

EPISODES = 1000


# 카트폴 예제에서의 액터-크리틱(A2C) 에이전트
class A2CAgent:
    def __init__(self, state_size, action_size, max_action):
        self.render = False
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action
        self.value_size = 1

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        # 정책신경망과 가치신경망 생성
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_optimizer = Adam(learning_rate=self.actor_lr, clipnorm=5.0)
        self.critic_optimizer = Adam(learning_rate=self.critic_lr, clipnorm=5.0)

        # self.model = self.build_actor_critic()
        # self.optimizer = Adam(learning_rate=1e-3, clipnorm=5.0)

        if self.load_model:
            self.actor.load_weights("reinforcement-learning-kr-master/2-cartpole/3-continuius-actor-critic/save_model/cartpole_actor_trained.h5")
            self.critic.load_weights("reinforcement-learning-kr-master/2-cartpole/3-continuius-actor-critic/save_model/cartpole_critic_trained.h5")

    # actor: 상태를 받아 각 행동의 확률을 계산
    def build_actor(self):
        input = tf.keras.Input(shape=(self.action_size,))
        dense = Dense(24, activation='tanh', kernel_initializer='he_uniform')(input)
        actor_mu = Dense(self.action_size, kernel_initializer='he_uniform')(dense)
        actor_sigma = Dense(self.action_size, activation='sigmoid', kernel_initializer='he_uniform')(dense)
        
        actor = tf.keras.Model(inputs=input, outputs=[actor_mu, actor_sigma])
        
        return actor

    # critic: 상태를 받아서 상태의 가치를 계산
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear', kernel_initializer='he_uniform'))
    
        return critic

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        mu, sigma = self.actor.predict(state, verbose=0)
        dist = tfp.distributions.Normal(loc=mu[0], scale=sigma[0])
        action = dist.sample([1])[0]
        action = np.clip(action, -self.max_action, self.max_action)
        return action

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, state, action, reward, next_state, done):
        with tf.GradientTape() as tape1:
            value = self.critic(state)[0]
            next_value = self.critic(next_state)[0]
            target = reward + (1-done) * self.discount_factor * next_value
            critic_loss = tf.reduce_mean(tf.square(target - value))
        grads = tape1.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape2:
            mu, sigma = self.actor(state)
            dist = tfp.distributions.Normal(loc=mu, scale=sigma)
            action_prob = dist.prob([action])[0]
            advantage = target - value
            actor_loss = tf.reduce_mean(-tf.math.log(action_prob + 1e-5) * advantage)
        grads = tape2.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

if __name__ == "__main__":
    env = gym.make('CartPoleContinuous-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # 액터-크리틱(A2C) 에이전트 생성
    agent = A2CAgent(state_size, action_size, max_action)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # 에피소드가 중간에 끝나면 -100 보상
            reward = reward if not done or score == 499 else -100

            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
                score = score if score == 500.0 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("reinforcement-learning-kr-master/2-cartpole/3-continuius-actor-critic/save_graph/cartpole_a2c.png")
                print("episode:", e, "  score:", score)

                # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    agent.actor.save_weights("reinforcement-learning-kr-master/2-cartpole/3-continuius-actor-critic/save_model/cartpole_actor.h5")
                    agent.critic.save_weights("reinforcement-learning-kr-master/2-cartpole/3-continuius-actor-critic/save_model/cartpole_critic.h5")
                    sys.exit()
