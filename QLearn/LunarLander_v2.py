import argparse
import numpy as np
import gymnasium as gym
from classes import Strategy, QTable
from enum import Enum


# ゲームの名称
GAME_NAME = 'LunarLander-v2'

# 行動の種類
class Action(Enum):
    DO_NOTHING = 0 # 何もしない
    FIRE_LEFT_ENGINE = 1 # 左エンジンを吹かす
    FIRE_MAIN_ENGINE = 2 # メインエンジンを吹かす
    FIRE_RIGHT_ENGINE = 3 # 右エンジンを吹かす
    UNDEFINED = 4 # 未定義状態. 念のため設定しておく
ACTION_ID = { Action.DO_NOTHING:0, Action.FIRE_LEFT_ENGINE:1, Action.FIRE_MAIN_ENGINE:2, Action.FIRE_RIGHT_ENGINE:3 } # 行動名から行動番号への変換表
N_ACTIONS = len(ACTION_ID) # 種類数

# 行動名の取得
def get_action_name(action: Action):
    if action == Action.DO_NOTHING:
        return 'DO_NOTHING'
    elif action == Action.FIRE_LEFT_ENGINE:
        return 'FIRE_LEFT_ENGINE'
    elif action == Action.FIRE_MAIN_ENGINE:
        return 'FIRE_MAIN_ENGINE'
    elif action == Action.FIRE_RIGHT_ENGINE:
        return 'FIRE_RIGHT_ENGINE'
    else:
        return 'UNDEFINED'

# Q学習の設定値（これらの設定値が妥当だとは限らない）
EPS = 0.1 # ε-greedyにおけるε
LEARNING_RATE = 0.1 # 学習率
DISCOUNT_FACTOR = 0.9 # 割引率

# Q学習用のQテーブル用
q_table = QTable(action_class=Action)


### 状態の定義 ###

def get_state(observation):

    # 常に「状態0」
    # これでは当然まともに学習されないので，適切に定義し直す必要がある
    return 0


### 関数 ###

# 行動戦略
def select_action(state, strategy: Strategy):
    global q_table

    # Q値最大行動を選択する戦略
    if strategy == Strategy.QMAX:
        return q_table.get_best_action(state)

    # ε-greedy
    elif strategy == Strategy.E_GREEDY:
        if np.random.rand() < EPS:
            return select_action(state, strategy=Strategy.RANDOM)
        else:
            return q_table.get_best_action(state)

    # ランダム戦略
    else:
        return np.random.choice([Action.DO_NOTHING, Action.FIRE_LEFT_ENGINE, Action.FIRE_MAIN_ENGINE, Action.FIRE_RIGHT_ENGINE])


### ここから処理開始 ###

def main():
    parser = argparse.ArgumentParser(description='OpenAI Gym LunarLander-v2')
    parser.add_argument('--games', type=int, default=1, help='num. of games to play')
    parser.add_argument('--max_steps', type=int, default=200, help='max num. of steps per game')
    parser.add_argument('--load', type=str, default='', help='filename of Q table to be loaded before learning')
    parser.add_argument('--save', type=str, default='', help='filename where Q table will be saved after learning')
    parser.add_argument('--testmode', help='this option runs the program without learning', action='store_true')
    parser.add_argument('--randmode', help='this option runs the program with random strategy', action='store_true')
    args = parser.parse_args()

    # ゲームの実行回数
    N_GAMES = args.games + 1

    # 1ゲームあたりのステップ数の最大値（これを超えた場合は強制的にゲーム打ち切り）
    MAX_STEPS = args.max_steps

    # ゲーム環境を作成
    env = gym.make(GAME_NAME, render_mode='human')

    # Qテーブルをロード
    if args.load != '':
        q_table.load(args.load)

    # ゲームを N_GAMES 回実行
    for game_ID in range(1, N_GAMES):

        print('Game {0} start.'.format(game_ID))

        # まず，ゲームを初期化
        observation, info = env.reset()
        state = get_state(observation)
        total_reward = 0

        # 最大 MAX_STEPS 分を for ループで実行
        for t in range(MAX_STEPS):

            # AIの行動をランダム選択
            if args.randmode:
                action = select_action(state, Strategy.RANDOM)
            elif args.testmode:
                action = select_action(state, Strategy.QMAX)
            else:
                action = select_action(state, Strategy.E_GREEDY)

            # 使わないかもしれないが，行動前の状態を別変数に退避
            prev_state = state

            # 選択した行動を実行．戻り値の意味は次の通り
            #   - observation, state: 行動後の観測量および状態
            #   - reward: 報酬
            #   - done: 終了フラグ
            #   - truncated, info: このプログラムでは使用しない
            observation, reward, done, truncated, info = env.step(ACTION_ID[action])
            state = get_state(observation)
            total_reward += reward

            # Qテーブルを更新
            if not (args.testmode or args.randmode):
                _, V = q_table.get_best_action(state, with_value=True)
                Q = q_table.get_Q_value(prev_state, action) # 現在のQ値
                Q = (1 - LEARNING_RATE) * Q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * V) # 新しいQ値
                q_table.set_Q_value(prev_state, action, Q) # 新しいQ値を登録

            # MAX_STEPS分が完了する前にゲームが終了した場合は，すぐに次のゲームに移行
            if done:
                break

        print('Game {0} end.'.format(game_ID))
        print('total reward: {0}'.format(total_reward))
        print('')

    # ゲームを閉じる
    env.close()

    # Qテーブルをセーブ
    if args.save != '':
        q_table.save(args.save)


if __name__ == '__main__':
    main()
