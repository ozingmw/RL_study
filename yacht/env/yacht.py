from typing import Optional
import gym
from gym import spaces
import numpy as np
import os

class Board(object):
    '''
    board
        board_key_value:
            보드 이름, 해당 숫자 변환을 위해 설정

        board_state:
            ONES ~ TOTAL까지 저장하는 dict
        
        board_able_state:
            점수 먹으면 더이상 그 점수 못먹게 삭제하기 위해 만든 리스트

        함수:
            reset(self):
                보드판 리셋
            get_able_key(self):
                self.able_state 리턴함수
            remove_choice_number(self, number: int):
                self.able_state에서 얻은 점수 삭제
            board_num_to_str(number: int) -> str:
                number에서 str 변환
    '''
    def __init__(self):
        self.board_state = {
            'ACES': 0,
            'DEUES': 0,
            'THREES': 0, 
            'FOURS': 0, 
            'FIVES': 0, 
            'SIXES': 0, 
            'BOUNS': 0, 
            'CHOICE': 0, 
            '4_OF_A_KIND': 0, 
            'FULL_HOUSE': 0, 
            'SMALL_STRAIGHT': 0, 
            'LARGE_STRAIGHT': 0, 
            'YACHT': 0,
            'TOTAL': 0,
        }

        self.able_state = [
            'ACES', 'DEUES', 'THREES', 'FOURS', 'FIVES', 'SIXES', 'CHOICE', '4_OF_A_KIND', 'FULL_HOUSE', 'SMALL_STRAIGHT', 'LARGE_STRAIGHT', 'YACHT'
        ]

    def reset(self):
        self.__init__()
    
    def get_able_key(self) -> list:
        return self.able_state
    
    def remove_choice_number(self, number: int):
        self.able_state.remove(Board.board_num_to_str(number))

    def board_num_to_str(number: int) -> str:
        board_key_value = {
            1: 'ACES', 2: 'DEUES', 3: 'THREES', 4: 'FOURS', 5: 'FIVES', 6: 'SIXES', 7: 'CHOICE', 8: '4_OF_A_KIND', 9: 'FULL_HOUSE', 10: 'SMALL_STRAIGHT', 11: 'LARGE_STRAIGHT', 12: 'YACHT'
        }
        return board_key_value[number]

class YachtEnv(gym.Env):
    '''
    Action_space:
        action[0]: REROLL
            0이면 리롤합니다.
        action[1]: CHOOSE_NUMBER
            주사위를 보고 점수를 골라 넣습니다. (위에서부터 1-13)
        action[2]: SELECT_DICE
            주사위를 선택합니다. REROLL == 1이면 모든 주사위를 선택합니다.

    Observation_space:
        GAME_BOARD: 게임판,
        DICE_STATE: 주사위 상태,
        REROLL_COUNT: 리롤 가능 횟수

    Action:
        {reroll: 1, choose_number: 10, reroll_dice_number: [1,3]}
    State(return):
        {DICE_1: 1, DICE_2: 3, DICE_3: 6, DICE_4: 1, DICE_5: 2, reroll_count: 2}

    EX)
        [0, 10, [1,3,4]]:
            선택한(1,3,4) 주사위 리롤함
        [1, 7, [2,3,5]]:
            7번(CHOICE)로 점수 획득 (주사위 선택 필요없음)
    '''
    def __init__(self):
        self.reroll_count = 2
        self.dice = np.zeros(5, dtype=np.int32)
        self.a = Board()
        self.b = Board()
        self.turn = self.a
        self.turn_count = 0

        self.action_space = spaces.Dict({
            'REROLL': spaces.Discrete(2),
            'CHOOSE_NUMBER': spaces.Discrete(12),
            'SELECT_DICE': spaces.Tuple([spaces.Discrete(6, start=1), spaces.Discrete(6, start=1), spaces.Discrete(6, start=1), spaces.Discrete(6, start=1), spaces.Discrete(6, start=1)]),
        })
        
        self.observation_space = spaces.Dict({
            'GAME_BOARD': spaces.Dict({
                'ACES': spaces.Discrete(6),
                'DEUES': spaces.Discrete(11),
                'THREES': spaces.Discrete(16),
                'FOURS': spaces.Discrete(21),
                'FIVES': spaces.Discrete(26),
                'SIXES': spaces.Discrete(31),
                'BOUNS': spaces.Discrete(35),            # 0 or 35 (63점 이상일때)
                'CHOICE': spaces.Discrete(31),
                '4_OF_A_KIND': spaces.Discrete(31),
                'FULL_HOUSE': spaces.Discrete(31),
                'SMALL_STRAIGHT': spaces.Discrete(15),   # 0 or 15
                'LARGE_STRAIGHT': spaces.Discrete(30),   # 0 or 30
                'YACHT': spaces.Discrete(50),            # 0 or 50
                # 총 점수
                'TOTAL': spaces.Discrete(280+1)
                # 최대 점수:
                # 6+12+18+24+30+35+30+30+15+30+50 = 280
            }),
            'DICE_STATE': spaces.Dict({
                'DICE_1': spaces.Discrete(6, start=1),
                'DICE_2': spaces.Discrete(6, start=1),
                'DICE_3': spaces.Discrete(6, start=1),
                'DICE_4': spaces.Discrete(6, start=1),
                'DICE_5': spaces.Discrete(6, start=1),
            }),
            'REROLL_COUNT': spaces.Discrete(3),
        })

    def step(self, action):
        if type(action[1]) == list:
            action[1] = list(map(int, action[1]))
            if self.reroll_count == 0:
                print('No remain reroll')
                return self.state, 0, False, (self.turn.board_state, {'Reroll_count': self.reroll_count}, {'Able_state': self.turn.get_able_key()})
            self._reroll(action[1])
            self.state = self.dice
            self.reroll_count -= 1
            return self.state, 0, False, (self.turn.board_state, {'Reroll_count': self.reroll_count}, {'Able_state': self.turn.get_able_key()})

        if Board.board_num_to_str(action[1]) not in self.turn.able_state:
            print('Number Error!')
            return self.state, 0, False, (self.turn.board_state, {'Reroll_count': self.reroll_count}, {'Able_state': self.turn.get_able_key()})

        score = self._calculate(action[1])
        self.turn.board_state[Board.board_num_to_str(action[1])] = score
        self.turn.remove_choice_number(action[1])
        self._check_over_63()
        self._check_total()
        reward = self.turn.board_state['TOTAL']
        info = (self.turn.board_state, {'Reroll_count': self.reroll_count}, {'Able_state': self.turn.get_able_key()})
        done = False
        self.turn_count += 1
        if self.turn_count >= 24:
            done = True
            total_a = self.a.board_state['TOTAL']
            total_b = self.b.board_state['TOTAL']
            self.print_board_state()
            if total_a == total_b:
                print('DRAW!')
            print('A IS THE WINNER!') if total_a > total_b else print('B IS THE WINNER!')
            return self.state, reward, done, info
        self._change_turn()
        return self.state, reward, done, info

    def reset(self):
        super().reset()
        self.a.reset()
        self.b.reset()
        self._reroll([1,2,3,4,5])
        self.state = self.dice
        return self.state

    def render(self, mode='ascii'):
        os.system('cls')
        self.print_board_state()

    def print_board_state(self):
        print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
        print('|\t\t    A\t\t\t|\t\t    B\t\t\t|')
        print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ|')
        print(f'| 1.\tACES\t\t\t\b\b| {self.a.board_state["ACES"]} {self._expect_score(1, self.a)}\t', end="")
        print(f'| 1.\tACES\t\t\t\b\b| {self.b.board_state["ACES"]} {self._expect_score(1, self.b)}\t|')
        print(f'| 2.\tDEUES\t\t\t\b\b| {self.a.board_state["DEUES"]} {self._expect_score(2, self.a)}\t', end="")
        print(f'| 2.\tDEUES\t\t\t\b\b| {self.b.board_state["DEUES"]} {self._expect_score(2, self.b)}\t|')
        print(f'| 3.\tTHREES\t\t\t\b\b| {self.a.board_state["THREES"]} {self._expect_score(3, self.a)}\t', end="")
        print(f'| 3.\tTHREES\t\t\t\b\b| {self.b.board_state["THREES"]} {self._expect_score(3, self.b)}\t|')
        print(f'| 4.\tFOURS\t\t\t\b\b| {self.a.board_state["FOURS"]} {self._expect_score(4, self.a)}\t', end="")
        print(f'| 4.\tFOURS\t\t\t\b\b| {self.b.board_state["FOURS"]} {self._expect_score(4, self.b)}\t|')
        print(f'| 5.\tFIVES\t\t\t\b\b| {self.a.board_state["FIVES"]} {self._expect_score(5, self.a)}\t', end="")
        print(f'| 5.\tFIVES\t\t\t\b\b| {self.b.board_state["FIVES"]} {self._expect_score(5, self.b)}\t|')
        print(f'| 6.\tSIXES\t\t\t\b\b| {self.a.board_state["SIXES"]} {self._expect_score(6, self.a)}\t', end="")
        print(f'| 6.\tSIXES\t\t\t\b\b| {self.b.board_state["SIXES"]} {self._expect_score(6, self.b)}\t|')
        print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ|', end="")
        print('\t\t\tDICE_1\tDICE_2\tDICE_3\tDICE_4\tDICE_5\tREROLL_COUNT')
        print(f'|\tBOUNS\t\t\t\b\b| {self.a.board_state["BOUNS"]} ({self._get_1_to_6()}/63)\t', end="")
        print(f'|\tBOUNS\t\t\t\b\b| {self.b.board_state["BOUNS"]}\t|', end="")
        print(f'\t\t\t  {self.dice[0]}\t  {self.dice[1]}\t  {self.dice[2]}\t  {self.dice[3]}\t  {self.dice[4]}\t     {self.reroll_count}')
        print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ|')
        print(f'| 7.\tCHOICE\t\t\t\b\b| {self.a.board_state["CHOICE"]} {self._expect_score(7, self.a)}\t', end="")
        print(f'| 7.\tCHOICE\t\t\t\b\b| {self.b.board_state["CHOICE"]} {self._expect_score(7, self.b)}\t|')
        print(f'| 8.\t4_OF_A_KIND\t\t\b\b| {self.a.board_state["4_OF_A_KIND"]} {self._expect_score(8, self.a)}\t', end="")
        print(f'| 8.\t4_OF_A_KIND\t\t\b\b| {self.b.board_state["4_OF_A_KIND"]} {self._expect_score(8, self.b)}\t|')
        print(f'| 9.\tFULL_HOUSE\t\t\b\b| {self.a.board_state["FULL_HOUSE"]} {self._expect_score(9, self.a)}\t', end="")
        print(f'| 9.\tFULL_HOUSE\t\t\b\b| {self.b.board_state["FULL_HOUSE"]} {self._expect_score(9, self.b)}\t|')
        print(f'| 10.\tSMALL_STRAIGHT\t\t\b\b| {self.a.board_state["SMALL_STRAIGHT"]} {self._expect_score(10, self.a)}\t', end="")
        print(f'| 10.\tSMALL_STRAIGHT\t\t\b\b| {self.b.board_state["SMALL_STRAIGHT"]} {self._expect_score(10, self.b)}\t|')
        print(f'| 11.\tLARGE_STRAIGHT\t\t\b\b| {self.a.board_state["LARGE_STRAIGHT"]} {self._expect_score(11, self.a)}\t', end="")
        print(f'| 11.\tLARGE_STRAIGHT\t\t\b\b| {self.b.board_state["LARGE_STRAIGHT"]} {self._expect_score(11, self.b)}\t|')
        print(f'| 12.\tYACHT\t\t\t\b\b| {self.a.board_state["YACHT"]} {self._expect_score(12, self.a)}\t', end="")
        print(f'| 12.\tYACHT\t\t\t\b\b| {self.b.board_state["YACHT"]} {self._expect_score(12, self.b)}\t|')
        print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ|')
        print(f'|\tTOTAL\t\t\t\b\b| {self.a.board_state["TOTAL"]}\t', end="")
        print(f'|\tTOTAL\t\t\t\b\b| {self.b.board_state["TOTAL"]}\t|')
        print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')

    def _expect_score(self, number, turn):
        if Board.board_num_to_str(number) not in self.turn.able_state:
            return ""
        return f"({self._calculate(number)})" if self.turn == turn else ""

    def _reroll(self, dice_list):
        for index in dice_list:
            self.dice[index-1] = np.random.randint(1, 7)
    
    def _calculate(self, number):
        score = 0
        if number == 1:
            score = sum(np.where(self.dice == 1, 1, 0))
        elif number == 2:
            score = sum(np.where(self.dice == 2, 2, 0))
        elif number == 3:
            score = sum(np.where(self.dice == 3, 3, 0))
        elif number == 4:
            score = sum(np.where(self.dice == 4, 4, 0))
        elif number == 5:
            score = sum(np.where(self.dice == 5, 5, 0))
        elif number == 6:
            score = sum(np.where(self.dice == 6, 6, 0))
        elif number == 7:
            score = np.sum(self.dice)
        elif number == 8:
            for num in range(1, 7):
                temp = np.where(self.dice != num)[0]
                if len(temp) == 1:
                    score = num * 4 + int(self.dice[temp])
                elif len(temp) == 0:
                    score = num * 5
        elif number == 9:
            for num1 in range(1, 7):
                temp = np.where(self.dice == num1)[0]
                if len(temp) == 5 and 'YACHT' not in self.turn.able_state:
                    score = num1 * 6
                elif len(temp) == 3:
                    for num2 in range(1, 7):
                        if num2 == num1:
                            continue
                        temp = np.where(self.dice == num2)[0]
                        if len(temp) == 2:
                            score = num1 * 3 + num2 * 2                        
        elif number == 10:
            temp = np.unique(self.dice)
            if len(temp) == 4:
                if (temp == [1,2,3,4]).all() or (temp == [2,3,4,5]).all() or (temp == [3,4,5,6]).all():
                    score = 15
        elif number == 11:
            if (np.sort(self.dice) == [1,2,3,4,5]).all() or (np.sort(self.dice) == [2,3,4,5,6]).all():
                score = 30
        elif number == 12:
            for num in range(1, 7):
                temp = np.where(self.dice != num)[0]
                if len(temp) == 0:
                    score = 50
        return score

    def _get_1_to_6(self, turn):
        if turn == 'a':
            return sum([self.a.board_state[Board.board_num_to_str(number)] for number in range(1, 7)])
            

    def _check_over_63(self):
        if self._get_1_to_6() >= 63:
            self.turn.board_state['BOUNS'] = 35

    def _check_total(self):
        total = sum([self.turn.board_state[Board.board_num_to_str(number)] for number in range(1, 13)])
        total += self.turn.board_state['BOUNS']
        self.turn.board_state['TOTAL'] = total

    def _change_turn(self):
        self.turn = self.b if self.turn == self.a else self.a
        self._reroll([1,2,3,4,5])
        self.reroll_count = 2
        print("Change Turn!")