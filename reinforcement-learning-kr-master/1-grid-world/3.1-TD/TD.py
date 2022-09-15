import numpy as np
import random
from collections import defaultdict
from environment import Env


# 몬테카를로 에이전트 (모든 에피소드 각각의 샘플로 부터 학습)
class MCAgent:
    def __init__(self, actions):
        self.width = 5
        self.height = 5
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        # self.samples = []
        self.value_table = defaultdict(float)
        self.state = str([0, 0])

    def reset_state(self):
        self.state = str([0, 0])

    # 메모리에 샘플을 추가
    # 즉시 업데이트하기 때문에 메모리 필요 없음
    # def save_sample(self, state, reward, done):
    #     self.samples.append([state, reward, done])

    # 현재 에피소드에서 다음 에피소드로 진행한 후 현재 에피소드에 대한 가치함수 업데이트
    def update(self, reward, next_state):
        next_state = str(next_state)
        self.value_table[self.state] = self.value_table[self.state] + self.learning_rate * (reward + self.discount_factor * self.value_table[next_state] - self.value_table[self.state])
        # print(self.value_table[self.state])
        self.state = next_state

    # 큐 함수에 따라서 행동을 반환
    # 입실론 탐욕 정책에 따라서 행동을 반환
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 랜덤 행동
            action = np.random.choice(self.actions)
        else:
            # 큐 함수에 따른 행동
            next_state = self.possible_next_state(state)
            action = self.arg_max(next_state)
        return int(action)

    # 후보가 여럿이면 arg_max를 계산하고 무작위로 하나를 반환
    @staticmethod
    def arg_max(next_state):
        max_index_list = []
        max_value = next_state[0]
        for index, value in enumerate(next_state):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

    # 가능한 다음 모든 상태들을 반환
    def possible_next_state(self, state):
        col, row = state
        next_state = [0.0] * 4

        if row != 0:
            next_state[0] = self.value_table[str([col, row - 1])]
        else:
            next_state[0] = self.value_table[str(state)]
        if row != self.height - 1:
            next_state[1] = self.value_table[str([col, row + 1])]
        else:
            next_state[1] = self.value_table[str(state)]
        if col != 0:
            next_state[2] = self.value_table[str([col - 1, row])]
        else:
            next_state[2] = self.value_table[str(state)]
        if col != self.width - 1:
            next_state[3] = self.value_table[str([col + 1, row])]
        else:
            next_state[3] = self.value_table[str(state)]

        return next_state


# 메인 함수
if __name__ == "__main__":
    env = Env()
    agent = MCAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        state = env.reset()
        action = agent.get_action(state)
        
        while True:
            env.render()

            # 다음 상태로 이동
            # 보상은 숫자이고, 완료 여부는 boolean
            # print(f'action: {env.action_space[action]}\n\n')
            next_state, reward, done = env.step(action)
            # print(next_state)
            # agent.save_sample(next_state, reward, done)

            # 행동한 상태 업데이트
            agent.update(reward, next_state)

            # 다음 행동 받아옴
            action = agent.get_action(next_state)

            # 에피소드가 완료
            if done:
                # agent.samples.clear()
                agent.reset_state()
                break
