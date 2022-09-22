import gym
import argparse
import numpy as np
from qtable import Strategy, QTable


# ゲームの名称
GAME_NAME = 'CartPole-v0'

# 行動の種類数
# 台車を「右」へ動かすか「左」へ動かすかの2種類
N_ACTIONS = 2

# Q学習の設定値（これらの設定値が妥当だとは限らない）
EPS = 0.1 # ε-greedyにおけるε
LEARNING_RATE = 0.1 # 学習率
DISCOUNT_FACTOR = 0.9 # 割引率

# Q学習用のQテーブル用
q_table = QTable(n_actions=N_ACTIONS)


### 状態の定義 ###

'''
CartPole-v0の観測量は, 以下の4つを要素とする4次元ベクトルと定義されている.
  - 1次元目: 台車の位置（-2.4 ~ 2.4）
  - 2次元目: 台車の速度（-inf ~ inf）
  - 3次元目: 棒の角度（-0.21 ~ 0.21）
  - 4次元目: 棒の先端の速度（-inf ～ inf）

Q学習を実行するためには, 上記の4次元ベクトルを適当に離散化し, 有限種類の「状態」に落とし込む必要がある.
'''
def get_state(observation):

    # 観測量の離散化
    cart_pos = np.digitize(observation[0], bins=[-2.4, -1.2, 0, 1.2, 2.4]) # 1次元目
    cart_vel = np.digitize(observation[1], bins=[-3.0, -1.5, 0, 1.5, 3.0]) # 2次元目
    pole_ang = np.digitize(observation[2], bins=[-0.2, -0.1, 0, 0.1, 0.2]) # 3次元目
    pole_vel = np.digitize(observation[3], bins=[-2.0, -1.0, 0, 1.0, 2.0]) # 4次元目

    # 離散化した観測値を tuple にして返却
    return (cart_pos, cart_vel, pole_ang, pole_vel)


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
        # 乱数を用いて 0 以上 n_actions 未満の整数をランダムに取得
        z = np.random.randint(0, N_ACTIONS)
        return z


### ここから処理開始 ###

parser = argparse.ArgumentParser(description='OpenAI Gym CartPole-v0')
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
env = gym.make(GAME_NAME)

# Qテーブルをロード
if args.load != '':
    q_table.load(args.load)

# ゲームを N_GAMES 回実行
for game_ID in range(1, N_GAMES):

    print('Game {0} start.'.format(game_ID))

    # まず，ゲームを初期化
    observation = env.reset()
    state = get_state(observation)
    total_reward = 0

    # 最大 MAX_STEPS 分を for ループで実行
    for t in range(MAX_STEPS):

        # 現在のゲーム画面をレンダリング
        env.render()

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
        observation, reward, done, _ = env.step(action)
        state = get_state(observation)
        total_reward += reward

        # Qテーブルを更新
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
