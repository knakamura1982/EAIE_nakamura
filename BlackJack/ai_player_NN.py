import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
sys.path.append(os.path.abspath('..'))
import copy
import socket
import argparse
import torch
import numpy as np
from classes import Action, Player, get_card_info, get_action_name
from config import PORT, BET, INITIAL_MONEY, N_DECKS
from NN_structure import BJNet
from mylib.utility import print_args


# 学習結果の保存先フォルダ
MODEL_DIR = './BJNet_models'


### グローバル変数 ###

# プレイヤークラスのインスタンスを作成
player = Player(initial_money=INITIAL_MONEY, basic_bet=BET)

# ディーラーとの通信用ソケット
soc = None

# ニューラルネットワーク用の変数の準備
nn_model = None

# 行動リスト
# ニューラルネットワークのクラス番号の順番に合わせて並べる（基本的にアルファベット順になるはず）
action_set = [Action.DOUBLE_DOWN, Action.HIT, Action.RETRY, Action.STAND, Action.SURRENDER]


### 関数 ###

# ゲームを開始する
def game_start(game_ID=0):
    global player, soc

    print('Game {0} start.'.format(game_ID))
    print('  money: ', player.get_money(), '$')

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
def select_action(state):
    global nn_model

    # 現状態を torch.tensor 型に変換
    x = torch.tensor(np.asarray([state]), dtype=torch.float32, device=DEVICE)

    # 現状態をニューラルネットワークに入力
    y = nn_model(x)
    y = torch.softmax(y, dim=1)
    z = y.to('cpu').detach().numpy()[0]

    # ニューラルネットワークの出力（行動選択確率）に応じて行動を選択
    return np.random.choice(a=action_set, size=1, p=z)


### ここから処理開始 ###

parser = argparse.ArgumentParser(description='AI Black Jack Player (Neural Network-based)')
parser.add_argument('--games', type=int, default=1, help='num. of games to play')
parser.add_argument('--history', type=str, default='play_log.csv', help='filename where game history will be saved')
parser.add_argument('--model', default=os.path.join(MODEL_DIR, 'model.pth'), type=str, help='file path of trained model')
parser.add_argument('--gpu', default=-1, type=int, help='GPU/CUDA ID (negative value indicates CPU)')
args = print_args(parser.parse_args())
DEVICE = args['device']
MODEL_PATH = args['model']

n_games = args['games'] + 1

# ログファイルを開く
logfile = open(args['history'], 'w')
print('score,hand_length,action,result,reward', file=logfile) # ログファイルにヘッダ行（項目名の行）を出力

# ニューラルネットワークの作成
nn_model = BJNet()
nn_model.load_state_dict(torch.load(MODEL_PATH))
nn_model = nn_model.to(DEVICE)
nn_model.eval()

# n_games回ゲームを実行
for n in range(1, n_games):

    # nゲーム目を開始
    game_start(n)

    # 「現在の状態」を取得
    state = get_state()

    while True:

        # 次に実行する行動を選択
        action = select_action(state) # このプログラムでは state によらずランダム選択なので，state の指定は実質無意味
        action_name = get_action_name(action) # 行動名を表す文字列を取得

        # 選択した行動を実際に実行
        # 戻り値:
        #   - done: 終了フラグ．今回の行動によりゲームが終了したか否か（終了した場合はTrue, 続行中ならFalse）
        #   - reward: 獲得金額（ゲーム続行中の場合は 0 , ただし RETRY を実行した場合は1回につき -BET/4 ）
        #   - status: 行動実行後のプレイヤーステータス（バーストしたか否か，勝ちか負けか，などの状態を表す文字列）
        reward, done, status = act(action)

        # 「現在の状態」を再取得
        prev_state = state # 使わないかもしれないが，行動前の状態を別変数に退避
        prev_score = prev_state[0] # 行動前のプレイヤー手札のスコア（prev_state の一つ目の要素）
        state = get_state()
        score = state[0] # 行動後のプレイヤー手札のスコア（state の一つ目の要素）

        # ログファイルに「行動前の状態」「行動の種類」「行動結果」「獲得金額」などの情報を記録
        print('{},{},{},{},{}'.format(prev_state[0], prev_state[1], action_name, status, reward), file=logfile)

        # 終了フラグが立った場合はnゲーム目を終了
        if done == True:
            break

    print('')

# ログファイルを閉じる
logfile.close()
