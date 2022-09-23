import copy
import socket
import argparse
import tkinter as tk
from classes import Player, Action, get_action_name
from config import PORT, BET, INITIAL_MONEY, MAX_CARDS_PER_GAME


class HumanPlayerWindow():

    def __init__(self, initial_money: int, basic_bet: int, max_cards_per_game: int):

        self.max_cards_per_game = max_cards_per_game

        # プレイヤークラスのインスタンスを作成
        self.player = Player(initial_money=initial_money, basic_bet=basic_bet)

        # ディーラーとの通信用ソケット
        self.soc = None

        # メインウィンドウ
        self.root = tk.Tk()
        self.root.title('Black Jack')
        self.root.geometry('{0}x440'.format(200+120*max_cards_per_game))
        self.root.protocol('WM_DELETE_WINDOW', (lambda: 'pass')())

        # 残額表示オブジェクト
        self.money_text = tk.StringVar()
        self.money_text.set("money: {0:5}$".format(initial_money))
        self.money_label = tk.Label(self.root, textvariable=self.money_text, font=('Arial', '12', 'bold'))
        self.money_label.place(x=-20+120*max_cards_per_game, y=10)

        # 勝敗表示オブジェクト
        self.result_text = tk.StringVar()
        self.result_text.set("")
        self.result_label = tk.Label(self.root, textvariable=self.result_text, font=('Arial', '24', 'bold'))
        self.result_label.place(x=330, y=210)

        # スコア表示用オブジェクト
        self.player_score_text = tk.StringVar()
        self.dealer_score_text = tk.StringVar()
        self.player_score_text.set("(score:   )")
        self.dealer_score_text.set("(score:   )")
        self.player_score_label = tk.Label(self.root, textvariable=self.player_score_text, font=('Arial', '14', 'bold'))
        self.dealer_score_label = tk.Label(self.root, textvariable=self.dealer_score_text, font=('Arial', '14', 'bold'))
        self.player_score_label.place(x=160, y=240)
        self.dealer_score_label.place(x=160, y=10)

        # カード表示用キャンバス
        self.empty_img = tk.PhotoImage(file="./imgs/0.png")
        self.dealer_label = tk.Label(text="Dealer's cards", font=('Arial', '14', 'bold'))
        self.player_label = tk.Label(text="Player's cards", font=('Arial', '14', 'bold'))
        self.dealer_label.place(x=10, y=10)
        self.player_label.place(x=10, y=240)
        self.dealer_canvas = [0] * max_cards_per_game
        self.player_canvas = [0] * max_cards_per_game
        self.dealer_canvas_img = [0] * max_cards_per_game
        self.player_canvas_img = [0] * max_cards_per_game
        for i in range(0, max_cards_per_game):
            self.dealer_canvas[i] = tk.Canvas(width=112, height=160)
            self.player_canvas[i] = tk.Canvas(width=112, height=160)
            self.dealer_canvas[i].photo = self.empty_img
            self.player_canvas[i].photo = self.empty_img
            self.dealer_canvas_img[i] = self.dealer_canvas[i].create_image(0, 0, image=self.dealer_canvas[i].photo, anchor=tk.NW)
            self.player_canvas_img[i] = self.player_canvas[i].create_image(0, 0, image=self.player_canvas[i].photo, anchor=tk.NW)
            self.dealer_canvas[i].place(x=10+120*i, y=40)
            self.player_canvas[i].place(x=10+120*i, y=270)

        # ボタン
        self.action_label = tk.Label(text="Action:", font=('Arial', '14', 'bold'))
        self.action_label.place(x=40+120*max_cards_per_game, y=160)
        self.start_button = tk.Button(width=14, text='Game Start', font=('Arial', '12', 'bold'), command=self.game_start)
        self.ht_button = tk.Button(width=15, text='HIT', font=('Arial', '12'), state=tk.DISABLED, command=lambda:self.step(Action.HIT))
        self.st_button = tk.Button(width=15, text='STAND', font=('Arial', '12'), state=tk.DISABLED, command=lambda:self.step(Action.STAND))
        self.dd_button = tk.Button(width=15, text='DOUBLE DOWN', font=('Arial', '12'), state=tk.DISABLED, command=lambda:self.step(Action.DOUBLE_DOWN))
        self.sr_button = tk.Button(width=15, text='SURRENDER', font=('Arial', '12'), state=tk.DISABLED, command=lambda:self.step(Action.SURRENDER))
        self.quit_button = tk.Button(width=14, text='Quit', font=('Arial', '12', 'bold'), command=self.game_quit)
        self.start_button.place(x=40+120*max_cards_per_game, y=50)
        self.ht_button.place(x=40+120*max_cards_per_game, y=190)
        self.st_button.place(x=40+120*max_cards_per_game, y=225)
        self.dd_button.place(x=40+120*max_cards_per_game, y=260)
        self.sr_button.place(x=40+120*max_cards_per_game, y=295)
        self.quit_button.place(x=40+120*max_cards_per_game, y=400)

    # 実行
    def run(self):
        self.root.mainloop()

    # プレイヤーカードの描画
    def draw_player_card(self, n: int, card: int):
        if card < 0:
            pc_img = tk.PhotoImage(file="./imgs/ura.png")
        else:
            pc_img = tk.PhotoImage(file="./imgs/{0}.png".format(card+1))
        self.player_canvas[n].photo = pc_img
        self.player_canvas[n].itemconfig(self.player_canvas_img[n], image=self.player_canvas[n].photo)

    # ディーラーカードの描画
    def draw_dealer_card(self, n: int, card: int):
        if card < 0:
            dc_img = tk.PhotoImage(file="./imgs/ura.png")
        else:
            dc_img = tk.PhotoImage(file="./imgs/{0}.png".format(card+1))
        self.dealer_canvas[n].photo = dc_img
        self.dealer_canvas[n].itemconfig(self.dealer_canvas_img[n], image=self.dealer_canvas[n].photo)

    # プレイヤーカードの描画状態を初期化
    def undraw_player_card(self, n: int):
        self.player_canvas[n].photo = self.empty_img
        self.player_canvas[n].itemconfig(self.player_canvas_img[n], image=self.player_canvas[n].photo)

    # ディーラーカードの描画状態を初期化
    def undraw_dealer_card(self, n: int):
        self.dealer_canvas[n].photo = self.empty_img
        self.dealer_canvas[n].itemconfig(self.dealer_canvas_img[n], image=self.dealer_canvas[n].photo)

    # 行動ボタンを有効化
    def activate_buttons(self):
        self.ht_button['state'] = tk.NORMAL
        self.st_button['state'] = tk.NORMAL
        self.dd_button['state'] = tk.NORMAL
        self.sr_button['state'] = tk.NORMAL
        self.start_button['state'] = tk.DISABLED
        self.quit_button['state'] = tk.DISABLED

    # 行動ボタンを無効化
    def deactivate_buttons(self):
        self.ht_button['state'] = tk.DISABLED
        self.st_button['state'] = tk.DISABLED
        self.dd_button['state'] = tk.DISABLED
        self.sr_button['state'] = tk.DISABLED
        self.start_button['state'] = tk.NORMAL
        self.quit_button['state'] = tk.NORMAL

    # ゲームを開始する
    def game_start(self):

        # 前回ゲームの結果表示を削除
        self.result_text.set("")

        # ディーラープログラムに接続する
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.connect((socket.gethostname(), PORT))
        self.soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # ベット
        bet, money = self.player.set_bet()
        self.money_text.set("money: {0:5}$  ( bet: {1:3}$ )".format(money, bet)) # 金額表示を更新

        # ディーラーから初期カード情報を受信
        dc, pc1, pc2 = self.player.receive_init_cards(self.soc)
        self.player_score_text.set("(score: {0})".format(self.player.get_score()))
        self.dealer_score_text.set("(score: {0})".format(self.player.get_dealer_score()))
        self.draw_dealer_card(0, dc)
        self.draw_dealer_card(1, -1)
        self.draw_player_card(0, pc1)
        self.draw_player_card(1, pc2)
        for i in range(2, self.max_cards_per_game):
            self.undraw_dealer_card(i)
            self.undraw_player_card(i)

        # 行動ボタンを有効化
        self.activate_buttons()

        # 「現在の状態」を取得
        self.state = self.get_state()

    # 現時点での手札情報（ディーラー手札は見えているもののみ）を取得
    def get_current_hands(self):
        return copy.deepcopy(self.player.player_hand), copy.deepcopy(self.player.dealer_hand)

    # HITを実行する
    def hit(self):

        # ディーラーにメッセージを送信
        self.player.send_message(self.soc, 'hit')

        # ディーラーから情報を受信
        pc, score, status, rate, dc = self.player.receive_message(dsoc=self.soc, get_player_card=True, get_dealer_cards=True)
        n = self.player.get_num_player_cards()
        self.draw_player_card(n-1, pc)
        self.player_score_text.set("(score: {0})".format(score))

        # バーストした場合はゲーム終了
        if status == 'bust':

            for i in range(len(dc)):
                self.draw_dealer_card(i+1, dc[i])
            self.dealer_score_text.set("(score: {0})".format(self.player.get_dealer_score()))

            # ディーラーとの通信をカット
            self.soc.close()

            # 行動ボタンを無効化
            self.deactivate_buttons()

            # 所持金額を更新
            reward = self.player.update_money(rate=rate)
            self.money_text.set("money: {0:5}$  ( bet: {1:3}$ )".format(self.player.get_money(), self.player.get_current_bet()))

            # 勝敗表示を更新
            self.result_text.set("Bust...")
            self.result_label['fg'] = 'blue'

            return reward, True, status

        # バーストしなかった場合は続行
        else:

            # 1ゲームで引けるカードの最大枚数に達してしまった場合，HITとDOUBLE DOWNを選択できないようにする
            if n >= self.max_cards_per_game:
                self.ht_button['state'] = tk.DISABLED
                self.dd_button['state'] = tk.DISABLED

            return 0, False, status

    # STANDを実行する
    def stand(self):

        # ディーラーにメッセージを送信
        self.player.send_message(self.soc, 'stand')

        # ディーラーから情報を受信
        score, status, rate, dc = self.player.receive_message(dsoc=self.soc, get_dealer_cards=True)

        # ディーラーのカードを画面に表示
        for i in range(len(dc)):
            self.draw_dealer_card(i+1, dc[i])
        self.player_score_text.set("(score: {0})".format(score))
        self.dealer_score_text.set("(score: {0})".format(self.player.get_dealer_score()))

        # ゲーム終了，ディーラーとの通信をカット
        self.soc.close()

        # 行動ボタンを無効化
        self.deactivate_buttons()

        # 所持金額を更新
        reward = self.player.update_money(rate=rate)
        self.money_text.set("money: {0:5}$  ( bet: {1:3}$ )".format(self.player.get_money(), self.player.get_current_bet()))

        # 勝敗表示を更新
        if status == 'lose':
            self.result_text.set("Lose...")
            self.result_label['fg'] = 'blue'
        elif status == 'win':
            self.result_text.set("Win!!")
            self.result_label['fg'] = 'red'
        else:
            self.result_text.set("Draw")
            self.result_label['fg'] = 'green'

        return reward, True, status

    # DOUBLE_DOWNを実行する
    def double_down(self):

        # 今回のみベットを倍にする
        bet, money = self.player.double_bet()
        self.money_text.set("money: {0:5}$  ( bet: {1:3}$ )".format(money, bet)) # 金額表示を更新

        # ディーラーにメッセージを送信
        self.player.send_message(self.soc, 'double_down')

        # ディーラーから情報を受信
        pc, score, status, rate, dc = self.player.receive_message(dsoc=self.soc, get_player_card=True, get_dealer_cards=True)
        n = self.player.get_num_player_cards()
        self.draw_player_card(n-1, pc)
        for i in range(len(dc)):
            self.draw_dealer_card(i+1, dc[i])
        self.player_score_text.set("(score: {0})".format(score))
        self.dealer_score_text.set("(score: {0})".format(self.player.get_dealer_score()))

        # ゲーム終了，ディーラーとの通信をカット
        self.soc.close()

        # 行動ボタンを無効化
        self.deactivate_buttons()

        # 所持金額を更新
        reward = self.player.update_money(rate=rate)
        self.money_text.set("money: {0:5}$  ( bet: {1:3}$ )".format(self.player.get_money(), self.player.get_current_bet()))

        # 勝敗表示を更新
        if status == 'bust':
            self.result_text.set("Bust...")
            self.result_label['fg'] = 'blue'
        elif status == 'lose':
            self.result_text.set("Lose...")
            self.result_label['fg'] = 'blue'
        elif status == 'win':
            self.result_text.set("Win!!")
            self.result_label['fg'] = 'red'
        else:
            self.result_text.set("Draw")
            self.result_label['fg'] = 'green'

        return reward, True, status

    # SURRENDERを実行する
    def surrender(self):

        # ディーラーにメッセージを送信
        self.player.send_message(self.soc, 'surrender')

        # ディーラーから情報を受信
        score, status, rate, dc = self.player.receive_message(dsoc=self.soc, get_dealer_cards=True)

        # ディーラーのカードを画面に表示
        for i in range(len(dc)):
            self.draw_dealer_card(i+1, dc[i])
        self.player_score_text.set("(score: {0})".format(score))
        self.dealer_score_text.set("(score: {0})".format(self.player.get_dealer_score()))

        # ゲーム終了，ディーラーとの通信をカット
        self.soc.close()

        # 行動ボタンを無効化
        self.deactivate_buttons()

        # 所持金額を更新
        reward = self.player.update_money(rate=rate)
        self.money_text.set("money: {0:5}$  ( bet: {1:3}$ )".format(self.player.get_money(), self.player.get_current_bet()))

        # 勝敗表示を更新
        self.result_text.set("Surrendered")
        self.result_label['fg'] = 'green'

        return reward, True, status

    # 行動の実行
    def act(self, action: Action):
        if action == Action.HIT:
            return self.hit()
        elif action == Action.STAND:
            return self.stand()
        elif action == Action.DOUBLE_DOWN:
            return self.double_down()
        elif action == Action.SURRENDER:
            return self.surrender()
        else:
            exit()

    # ゲームを終了する
    def game_quit(self):
        self.root.destroy()

    # 現在の状態の取得
    def get_state(self):

        # 現在の手札情報を取得
        # #   - p_hand: プレイヤー手札
        # #   - d_hand: ディーラー手札（見えているもののみ）
        p_hand, d_hand = self.get_current_hands()

        # 「現在の状態」を設定
        # ここでは例として，プレイヤー手札のスコアとプレイヤー手札の枚数の組を「現在の状態」とする
        score = p_hand.get_score() # プレイヤー手札のスコア
        length = p_hand.length() # プレイヤー手札の枚数
        state = (score, length) # 現在の状態

        return state

    # 1ステップ分ゲームを実行
    # 行動ボタンを押すと，この関数が実行される
    def step(self, action: Action):

        action_name = get_action_name(action) # 行動名を表す文字列を取得

        # 選択した行動を実際に実行
        # 戻り値:
        #   - done: 終了フラグ．今回の行動によりゲームが終了したか否か（終了した場合はTrue, 続行中ならFalse）
        #   - reward: 獲得金額（ゲーム続行中の場合は 0 ）
        #   - status: 行動実行後のプレイヤーステータス（バーストしたか否か，勝ちか負けか，などの状態を表す文字列）
        reward, done, status = self.act(action)

        # 「現在の状態」を再取得
        prev_state = self.state # 使わないかもしれないが，行動前の状態を別変数に退避
        prev_score = prev_state[0] # 行動前のプレイヤー手札のスコア（prev_state の一つ目の要素）
        state = self.state = self.get_state()
        score = state[0] # 行動後のプレイヤー手札のスコア（state の一つ目の要素）

        # ログファイルに「行動前の状態」「行動の種類」「行動結果」「獲得金額」などの情報を記録
        print('{},{},{},{},{}'.format(prev_state[0], prev_state[1], action_name, status, reward), file=logfile)


### ここから処理開始 ###

parser = argparse.ArgumentParser(description='Black Jack Client')
parser.add_argument('--history', type=str, default='play_log.csv', help='filename where game history will be saved')
args = parser.parse_args()

# ログファイルを開く
logfile = open(args.history, 'w')
print('score,hand_length,action,result,reward', file=logfile) # ログファイルにヘッダ行（項目名の行）を出力

# メインループ開始
window = HumanPlayerWindow(initial_money=INITIAL_MONEY, basic_bet=BET, max_cards_per_game=MAX_CARDS_PER_GAME)
window.run()

# ログファイルを閉じる
logfile.close()
