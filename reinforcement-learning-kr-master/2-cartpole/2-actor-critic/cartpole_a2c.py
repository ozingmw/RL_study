import sys
import gym
import pylab
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

import tensorflow as tf

EPISODES = 1000


# 카트폴 예제에서의 액터-크리틱(A2C) 에이전트
class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
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
            self.actor.load_weights("reinforcement-learning-kr-master/2-cartpole/2-actor-critic/save_model/cartpole_actor_trained.h5")
            self.critic.load_weights("reinforcement-learning-kr-master/2-cartpole/2-actor-critic/save_model/cartpole_critic_trained.h5")

    # actor: 상태를 받아 각 행동의 확률을 계산
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='tanh', kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform'))
        actor.summary()
        # actor.compile(loss='binary_cross_entropy', optimizer=self.actor_updater)
        return actor

    # critic: 상태를 받아서 상태의 가치를 계산
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear', kernel_initializer='he_uniform'))
        critic.summary()
        # critic.compile(loss='mse', optimizer=self.critic_updater)
        return critic

    # def build_actor_critic(self):
    #     input = tf.keras.Input(shape=(self.state_size,))
    #     x = tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform')(input)
    #     actor = tf.keras.layers.Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')(x)
    #     critic = tf.keras.layers.Dense(24, activation='relu', kernel_initializer='he_uniform')(x)
    #     critic = tf.keras.layers.Dense(self.value_size, activation='linear', kernel_initializer='he_uniform')(critic)

    #     model = tf.keras.Model(inputs=input, outputs=[actor, critic])
    #     model.summary()

    #     return model


    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        policy = self.actor.predict(state, verbose=0)[0]
        # policy, _ = self.model.predict(state, verbose=0)
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # 정책신경망을 업데이트하는 함수
    # def actor_optimizer(self):
    #     action = K.placeholder(shape=[None, self.action_size])
    #     advantage = K.placeholder(shape=[None, ])

    #     action_prob = K.sum(action * self.actor.output, axis=1)
    #     cross_entropy = K.log(action_prob) * advantage
    #     loss = -K.sum(cross_entropy)

    #     optimizer = Adam(lr=self.actor_lr)
    #     updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
    #     train = K.function([self.actor.input, action, advantage], [], updates=updates)
    #     return train

    # 가치신경망을 업데이트하는 함수
    # def critic_optimizer(self):
    #     target = K.placeholder(shape=[None, ])

    #     loss = K.mean(K.square(target - self.critic.output))

    #     optimizer = Adam(lr=self.critic_lr)
    #     updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
    #     train = K.function([self.critic.input, target], [], updates=updates)

    #     return train

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
            policy = self.actor(state)[0]
            mask = tf.one_hot([action], self.action_size)
            action_prob = tf.reduce_sum(policy * mask, axis=1)
            advantage = target - value
            actor_loss = tf.reduce_mean(-tf.math.log(action_prob + 1e-5) * advantage)
        grads = tape2.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        # with tf.GradientTape() as tape:
        #     policy, value = self.model(state)
        #     _, next_value = self.model(next_state)
        #     target = reward + (1-done) * self.discount_factor * next_value[0]
        #     critic_loss = tf.reduce_mean(0.5 * tf.square(target - value[0]))
        #     mask = tf.one_hot([action], self.action_size)
        #     action_prob = tf.reduce_sum(policy[0] * mask, axis=1)
        #     advantage = target - value[0]
        #     actor_loss = -tf.reduce_mean(tf.math.log(action_prob + 1e-5) * advantage)

        #     loss = 0.2 * actor_loss + critic_loss
        
        # grads = tape.gradient(loss, self.model.trainable_variables)
        # self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))



if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')
    # 환경으로부터 상태와 행동의 크기를 받아옴
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 액터-크리틱(A2C) 에이전트 생성
    agent = A2CAgent(state_size, action_size)

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
                pylab.savefig("reinforcement-learning-kr-master/2-cartpole/2-actor-critic/save_graph/cartpole_a2c.png")
                print("episode:", e, "  score:", score)

                # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    agent.actor.save_weights("reinforcement-learning-kr-master/2-cartpole/2-actor-critic/save_model/cartpole_actor.h5")
                    agent.critic.save_weights("reinforcement-learning-kr-master/2-cartpole/2-actor-critic/save_model/cartpole_critic.h5")
                    sys.exit()
