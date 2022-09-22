import pickle
import numpy as np
from enum import Enum


# 行動戦略の定義
class Strategy(Enum):
    UNDEFINED = 0
    RANDOM = 1
    QMAX = 2
    E_GREEDY = 3
    MANUAL = 4 # myGame.pyでのみ有効（それ以外のケースではRANDOMと解釈される）


# Q学習に用いるQテーブルを実現するクラス
class QTable:

    # コンストラクタ
    #   - default_value: まだ一度も試していない行動の初期Q値
    #   - n_actions: 行動の種類数
    def __init__(self, n_actions, default_value=0):
        self.n_actions = n_actions
        self.default_value = default_value
        self.table = {}

    # 状態 state の下で行動 action を実行する場合のQ値として value をセットする
    def set_Q_value(self, state, action: int, value):
        if 0 <= action and action < self.n_actions:
            key = (state, action)
            self.table[key] = value

    # 状態 state の下で行動 action を実行する場合のQ値
    def get_Q_value(self, state, action: int):
        key = (state, action)
        if key in self.table.keys():
            return self.table[key]
        else:
            return self.default_value

    # 状態 state の下でQ値が最大となる行動を取得（同じQ値を持つ行動が複数ある場合は，その中からランダム選択）
    #   - with_value: Trueの場合，Q値最大の行動とともに，そのときのQ値も返される
    def get_best_action(self, state, with_value=False):
        best_actions = []
        best_value = 0
        for a in range(self.n_actions):
            q = self.get_Q_value(state, a)
            if len(best_actions) == 0:
                best_actions.append(a)
                best_value = q
            else:
                if q < best_value:
                    continue
                elif q > best_value:
                    best_actions.clear()
                best_actions.append(a)
                best_value = q
        if with_value:
            return np.random.choice(best_actions), best_value
        else:
            return np.random.choice(best_actions)

    # Qテーブルをファイルに保存
    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self.table, f)

    # ファイルからQテーブルをロード
    def load(self, filename: str):
        with open(filename, 'rb') as f:
            self.table = pickle.load(f)
