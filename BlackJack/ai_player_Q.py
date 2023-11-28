import copy
import socket
import argparse
import numpy as np
from classes import Action, Strategy, QTable, Player, get_card_info, get_action_name
from config import PORT, BET, INITIAL_MONEY, N_DECKS


# 1ゲームあたりのRETRY回数の上限
RETRY_MAX = 10


### グローバル変数 ###

# ゲームごとのRETRY回数のカウンター
g_retry_counter = 0

# プレイヤークラスのインスタンスを作成
player = Player(initial_money=INITIAL_MONEY, basic_bet=BET)

# ディーラーとの通信用ソケット
soc = None

# Q学習用のQテーブル
q_table = QTable(action_class=Action, default_value=0)

# Q学習の設定値
EPS = 0.3 # ε-greedyにおけるε
LEARNING_RATE = 0.1 # 学習率
DISCOUNT_FACTOR = 0.9 # 割引率


### 関数 ###

# ゲームを開始する
def game_start(game_ID=0):
    global g_retry_counter, player, soc

    print('Game {0} start.'.format(game_ID))
    print('  money: ', player.get_money(), '$')

    # RETRY回数カウンターの初期化
    g_retry_counter = 0

    # ディーラープログラムに接続する
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((socket.gethostname(), PORT))
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # ベット
    bet, money = player.set_bet()
    print('Action: BET')
    print('  money: ', money, '$')
    print('  bet: ', bet, '$')

    # ディーラーから「カードシャッフルを行ったか否か」の情報を取得
    # シャッフルが行われた場合は True が, 行われなかった場合は False が，変数 cardset_shuffled にセットされる
    # なお，本サンプルコードではここで取得した情報は使用していない
    cardset_shuffled = player.receive_card_shuffle_status(soc)
    if cardset_shuffled:
        print('Dealer said: Card set has been shuffled before this game.')

    # ディーラーから初期カード情報を受信
    dc, pc1, pc2 = player.receive_init_cards(soc)
    print('Delaer gave cards.')
    print('  dealer-card: ', get_card_info(dc))
    print('  player-card 1: ', get_card_info(pc1))
    print('  player-card 2: ', get_card_info(pc2))
    print('  current score: ', player.get_score())

# 現時点での手札情報（ディーラー手札は見えているもののみ）を取得
def get_current_hands():
    return copy.deepcopy(player.player_hand), copy.deepcopy(player.dealer_hand)

# HITを実行する
def hit():
    global player, soc

    print('Action: HIT')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'hit')

    # ディーラーから情報を受信
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
    print('  player-card {0}: '.format(player.get_num_player_cards()), get_card_info(pc))
    print('  current score: ', score)

    # バーストした場合はゲーム終了
    if status == 'bust':
        for i in range(len(dc)):
            print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
        print("  dealer's score: ", player.get_dealer_score())
        soc.close() # ディーラーとの通信をカット
        reward = player.update_money(rate=rate) # 所持金額を更新
        print('Game finished.')
        print('  result: bust')
        print('  money: ', player.get_money(), '$')
        return reward, True, status

    # バーストしなかった場合は続行
    else:
        return 0, False, status

# STANDを実行する
def stand():
    global player, soc

    print('Action: STAND')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'stand')

    # ディーラーから情報を受信
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
    print('  current score: ', score)
    for i in range(len(dc)):
        print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
    print("  dealer's score: ", player.get_dealer_score())

    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # 所持金額を更新
    reward = player.update_money(rate=rate)
    print('Game finished.')
    print('  result: ', status)
    print('  money: ', player.get_money(), '$')
    return reward, True, status

# DOUBLE_DOWNを実行する
def double_down():
    global player, soc

    print('Action: DOUBLE DOWN')

    # 今回のみベットを倍にする
    bet, money = player.double_bet()
    print('  money: ', money, '$')
    print('  bet: ', bet, '$')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'double_down')

    # ディーラーから情報を受信
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
    print('  player-card {0}: '.format(player.get_num_player_cards()), get_card_info(pc))
    print('  current score: ', score)
    for i in range(len(dc)):
        print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
    print("  dealer's score: ", player.get_dealer_score())

    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # 所持金額を更新
    reward = player.update_money(rate=rate)
    print('Game finished.')
    print('  result: ', status)
    print('  money: ', player.get_money(), '$')
    return reward, True, status

# SURRENDERを実行する
def surrender():
    global player, soc

    print('Action: SURRENDER')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'surrender')

    # ディーラーから情報を受信
    score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)
    print('  current score: ', score)
    for i in range(len(dc)):
        print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
    print("  dealer's score: ", player.get_dealer_score())

    # ゲーム終了，ディーラーとの通信をカット
    soc.close()

    # 所持金額を更新
    reward = player.update_money(rate=rate)
    print('Game finished.')
    print('  result: ', status)
    print('  money: ', player.get_money(), '$')
    return reward, True, status

# RETRYを実行する
def retry():
    global player, soc

    print('Action: RETRY')

    # ベット額の 1/4 を消費
    penalty = player.current_bet // 4
    player.consume_money(penalty)
    print('  player-card {0} has been removed.'.format(player.get_num_player_cards()))
    print('  money: ', player.get_money(), '$')

    # ディーラーにメッセージを送信
    player.send_message(soc, 'retry')

    # ディーラーから情報を受信
    pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True, retry_mode=True)
    print('  player-card {0}: '.format(player.get_num_player_cards()), get_card_info(pc))
    print('  current score: ', score)

    # バーストした場合はゲーム終了
    if status == 'bust':
        for i in range(len(dc)):
            print('  dealer-card {0}: '.format(i+2), get_card_info(dc[i]))
        print("  dealer's score: ", player.get_dealer_score())
        soc.close() # ディーラーとの通信をカット
        reward = player.update_money(rate=rate) # 所持金額を更新
        print('Game finished.')
        print('  result: bust')
        print('  money: ', player.get_money(), '$')
        return reward-penalty, True, status

    # バーストしなかった場合は続行
    else:
        return -penalty, False, status

# 行動の実行
def act(action: Action):
    if action == Action.HIT:
        return hit()
    elif action == Action.STAND:
        return stand()
    elif action == Action.DOUBLE_DOWN:
        return double_down()
    elif action == Action.SURRENDER:
        return surrender()
    elif action == Action.RETRY:
        return retry()
    else:
        exit()


### これ以降の関数が重要 ###

# 現在の状態の取得
def get_state():

    # 現在の手札情報を取得
    #   - p_hand: プレイヤー手札
    #   - d_hand: ディーラー手札（見えているもののみ）
    p_hand, d_hand = get_current_hands()

    # 「現在の状態」を設定
    # ここでは例として，プレイヤー手札のスコアとプレイヤー手札の枚数の組を「現在の状態」とする
    score = p_hand.get_score() # プレイヤー手札のスコア
    length = p_hand.length() # プレイヤー手札の枚数
    state = (score, length) # 現在の状態

    return state

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
        z = np.random.randint(0, 4)
        if z == 0:
            return Action.HIT
        elif z == 1:
            return Action.STAND
        elif z == 2:
            return Action.DOUBLE_DOWN
        elif z == 3:
            return Action.SURRENDER
        else: # z == 4 のとき
            return Action.RETRY


### ここから処理開始 ###

parser = argparse.ArgumentParser(description='AI Black Jack Player (Q-learning)')
parser.add_argument('--games', type=int, default=1, help='num. of games to play')
parser.add_argument('--history', type=str, default='play_log.csv', help='filename where game history will be saved')
parser.add_argument('--load', type=str, default='', help='filename of Q table to be loaded before learning')
parser.add_argument('--save', type=str, default='', help='filename where Q table will be saved after learning')
parser.add_argument('--testmode', help='this option runs the program without learning', action='store_true')
args = parser.parse_args()

n_games = args.games + 1

# Qテーブルをロード
if args.load != '':
    q_table.load(args.load)

# ログファイルを開く
logfile = open(args.history, 'w')
print('score,hand_length,action,result,reward', file=logfile) # ログファイルにヘッダ行（項目名の行）を出力

# n_games回ゲームを実行
for n in range(1, n_games):

    # nゲーム目を開始
    game_start(n)

    # 「現在の状態」を取得
    state = get_state()

    while True:

        # 次に実行する行動を選択
        if args.testmode:
            action = select_action(state, Strategy.QMAX)
        else:
            action = select_action(state, Strategy.E_GREEDY)
        if g_retry_counter >= RETRY_MAX and action == Action.RETRY:
            # RETRY回数が上限に達しているにもかかわらずRETRYが選択された場合，他の行動をランダムに選択
            action = np.random.choice([
                Action.HIT, Action.STAND, Action.DOUBLE_DOWN, Action.SURRENDER
            ])
        action_name = get_action_name(action) # 行動名を表す文字列を取得

        # 選択した行動を実際に実行
        # 戻り値:
        #   - done: 終了フラグ．今回の行動によりゲームが終了したか否か（終了した場合はTrue, 続行中ならFalse）
        #   - reward: 獲得金額（ゲーム続行中の場合は 0 , ただし RETRY を実行した場合は1回につき -BET/4 ）
        #   - status: 行動実行後のプレイヤーステータス（バーストしたか否か，勝ちか負けか，などの状態を表す文字列）
        reward, done, status = act(action)

        # 実行した行動がRETRYだった場合はRETRY回数カウンターを1増やす
        if action == Action.RETRY:
            g_retry_counter += 1

        # 「現在の状態」を再取得
        prev_state = state # 行動前の状態を別変数に退避
        prev_score = prev_state[0] # 行動前のプレイヤー手札のスコア（prev_state の一つ目の要素）
        state = get_state()
        score = state[0] # 行動後のプレイヤー手札のスコア（state の一つ目の要素）

        # Qテーブルを更新
        if not args.testmode:
            _, V = q_table.get_best_action(state, with_value=True)
            Q = q_table.get_Q_value(prev_state, action) # 現在のQ値
            Q = (1 - LEARNING_RATE) * Q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * V) # 新しいQ値
            q_table.set_Q_value(prev_state, action, Q) # 新しいQ値を登録

        # ログファイルに「行動前の状態」「行動の種類」「行動結果」「獲得金額」などの情報を記録
        print('{},{},{},{},{}'.format(prev_state[0], prev_state[1], action_name, status, reward), file=logfile)

        # 終了フラグが立った場合はnゲーム目を終了
        if done == True:
            break

    print('')

# ログファイルを閉じる
logfile.close()

# Qテーブルをセーブ
if args.save != '':
    q_table.save(args.save)
