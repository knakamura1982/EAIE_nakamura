import pickle
import socket
import numpy as np
from enum import Enum


# プレイヤーの取り得る行動の定義
class Action(Enum):
    UNDEFINED = 0
    HIT = 1
    STAND = 2
    DOUBLE_DOWN = 3
    SURRENDER = 4


# 行動戦略の定義
class Strategy(Enum):
    UNDEFINED = 0
    RANDOM = 1
    QMAX = 2
    E_GREEDY = 3


# カードセット
class CardSet:

    def __init__(self, n_decks: int):
        self.n_cards = 52 * n_decks
        self.all_cards = np.arange(0, self.n_cards)
        self.pos = -1
        self.shuffle()

    # シャッフル
    def shuffle(self):
        np.random.shuffle(self.all_cards)
        self.pos = 0

    # 1枚ドロー
    def draw(self):
        card = self.all_cards[self.pos]
        self.pos += 1
        return card

    # 残りカード枚数を取得
    def remaining_cards(self):
        return self.n_cards - self.pos


# 手札
class Hand:

    def __init__(self):
        self.clear()

    # i枚目を取得
    def __getitem__(self, i):
        return self.cards[i]

    # 手札の枚数
    def __len__(self):
        return len(self.cards)
    def length(self):
        return len(self.cards)

    # カード c を追加
    def append(self, c):
        self.cards.append(c)

    # 手札をクリア（0枚にする）
    def clear(self):
        self.cards = []

    # 現在のスコアを計算
    def get_score(self):
        tmp = []
        have_ace = False
        for i in self.cards:
            j = min(10, (i % 13) + 1)
            if j != 1:
                tmp.append(j)
            else:
                if have_ace:
                    tmp.append(1)
                else:
                    have_ace = True
        score = sum(tmp)
        if have_ace:
            if score + 11 > 21:
                score += 1
            else:
                score += 11
        return score

    # ナチュラルブラックジャックか否かを判定
    def is_nbj(self):
        if self.get_score() == 21 and len(self.cards) == 2:
            return True
        else:
            return False

    # バーストか否かを判定
    def is_busted(self):
        if self.get_score() > 21:
            return True
        else:
            return False


# 行動名の取得
def get_action_name(action: Action):
    if action == Action.HIT:
        return 'HIT'
    elif action == Action.STAND:
        return 'STAND'
    elif action == Action.DOUBLE_DOWN:
        return 'DOUBLE DOWN'
    elif action == Action.SURRENDER:
        return 'SURRENDER'
    else:
        return 'UNDEFINED'


# カードのスート・数字を取得
def get_card_info(card):

    n = (card % 13) + 1
    if n == 1:
        num = 'A'
    elif n == 11:
        num = 'J'
    elif n == 12:
        num = 'Q'
    elif n == 13:
        num = 'K'
    else:
        num = '{0}'.format(n)

    s = card // 13
    if s == 0:
        suit = 'Spade'
    elif s == 1:
        suit = 'Club'
    elif s == 2:
        suit = 'Diamond'
    else:
        suit = 'Heart'

    return suit + '-' + num


# プレイヤークラス
class Player:

    # コンストラクタ
    #   - initial_money: 初期所持金
    #   - basic_bet: 基本ベット額
    def __init__(self, initial_money: int, basic_bet: int):

        # パラメータをメンバ変数にセット
        self.initial_money = initial_money
        self.basic_bet = basic_bet

        # 現在の所持金
        self.money = self.initial_money

        # 現在のベット額
        self.current_bet = 0

        # 手札
        self.player_hand = Hand() # プレイヤーの手札
        self.dealer_hand = Hand() # ディーラーの手札（見えているもののみ）

    # 現在のプレイヤースコアを取得
    def get_score(self):
        return self.player_hand.get_score()

    # 現在のディーラースコアを取得
    def get_dealer_score(self):
        return self.dealer_hand.get_score()

    # 現在の所持金額を取得
    def get_money(self):
        return self.money

    # 現在のベット額を取得
    def get_current_bet(self):
        return self.current_bet

    # 現在のプレイヤーカードの枚数を取得
    def get_num_player_cards(self):
        return len(self.player_hand.cards)

    # ベットの設定
    def set_bet(self):
        self.current_bet = self.basic_bet
        self.money -= self.current_bet
        return self.current_bet, self.money

    # ベットを倍額にする
    def double_bet(self):
        self.money -= self.current_bet
        self.current_bet *= 2
        return self.current_bet, self.money

    # 所持金額を更新
    # 戻り値: 獲得金額 = 更新後所持金 - 更新前所持金（Q学習において報酬として利用）
    def update_money(self, rate: float):
        refund = int(self.current_bet * rate)
        reward = refund - self.current_bet
        self.money += refund
        self.current_bet = 0
        return reward

    # ディーラーから初期カード情報を受信
    #   - dsoc: ディーラーとの間でのメッセージを送受信するためのソケット
    def receive_init_cards(self, dsoc: socket.socket):
        msg = dsoc.recv(1024).decode("utf-8").split(',')
        dc = int(msg[0])
        pc1 = int(msg[1])
        pc2 = int(msg[2])
        self.dealer_hand.clear()
        self.player_hand.clear()
        self.dealer_hand.append(dc)
        self.player_hand.append(pc1)
        self.player_hand.append(pc2)
        return dc, pc1, pc2

    # ディーラーへのメッセージ送信
    #   - dsoc: ディーラーとの間でのメッセージを送受信するためのソケット
    #   - msg: ディーラーに送信するメッセージテキスト
    def send_message(self, dsoc: socket.socket, msg: str):
        dsoc.send(bytes(msg, 'utf-8'))

    # ディーラーからのメッセージ受信
    #   - dsoc: ディーラーとの間でのメッセージを送受信するためのソケット
    #   - get_player_card: プレイヤーカードの種類を通知するか否か
    #   - get_dealer_cards: ディーラーの手札を通知するか否か
    def receive_message(self, dsoc: socket.socket, get_player_card=False, get_dealer_cards=False):
        msg = dsoc.recv(1024).decode("utf-8").split(',')
        if get_player_card:
            player_card = int(msg[0])
            self.player_hand.append(player_card)
            msg = msg[1:]
        status = msg[1]
        score = int(msg[0])
        rate = float(msg[2])
        if get_dealer_cards:
            dealer_cards = []
            for i in range(3, len(msg)):
                dc = int(msg[i])
                dealer_cards.append(dc)
                self.dealer_hand.append(dc)
            if get_player_card:
                return player_card, score, status, rate, dealer_cards
            else:
                return score, status, rate, dealer_cards
        else:
            if get_player_card:
                return player_card, score, status, rate
            else:
                return score, status, rate


# Q学習に用いるQテーブルを実現するクラス
class QTable:

    # コンストラクタ
    #   - default_value: まだ一度も試していない行動の初期Q値
    def __init__(self, default_value=0):
        self.default_value = default_value
        self.table = {}

    # 状態 state の下で行動 action を実行する場合のQ値として value をセットする
    def set_Q_value(self, state, action: Action, value):
        key = (state, action)
        self.table[key] = value

    # 状態 state の下で行動 action を実行する場合のQ値
    def get_Q_value(self, state, action: Action):
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
        for a in Action:
            if a == Action.UNDEFINED:
                continue
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
