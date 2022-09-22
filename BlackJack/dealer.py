import socket
from classes import Action, CardSet, Hand
from config import PORT, N_DECKS, SHUFFLE_INTERVAL, MAX_CARDS_PER_GAME


# ディーラークラス
class Dealer():

    # カードIDからスートと数字の情報のみを抽出するスタティックメソッド
    # 戻り値の整数を13で割ったときの商 a, 余りを b として，
    #   - a: スート（0: スペード，1:クラブ，2:ダイヤ，3:ハート）
    #   - b+1: 数字
    # となる
    @staticmethod
    def get_info(card):
        return card % 52

    # コンストラクタ
    #   - n_decks: 使用するデッキの数
    #   - shuffle_interval: カードシャッフルのあと何回続けてゲームをするか
    #   - max_cards_per_game: 1ゲームで引けるカードの最大枚数
    def __init__(self, n_decks: int, shuffle_interval: int, max_cards_per_game: int):

        # パラメータをメンバ変数にセット
        self.n_decks = n_decks
        self.shuffle_interval = shuffle_interval
        self.max_cards_per_game = max_cards_per_game

        # カードセットの用意
        self.card_set = CardSet(n_decks=self.n_decks)

        # 手札の用意
        self.dealer_hand = Hand() # ディーラーの手札
        self.player_hand = Hand() # プレイヤーの手札

        # ゲームID（現在が何回目のゲームかを示す変数）
        self.game_ID = 0

    # 勝敗判定の実行
    # 戻り値は (status, rate) のタプル
    #   - status: 勝敗ステータス（'win', 'lose', 'draw' のいずれか）
    #   - rate: 配当倍率
    def judge(self):

        # プレイヤーがバーストしている場合 => 負け
        if self.player_hand.is_busted():
            return 'lose', 0.0

        # プレイヤーがナチュラルブラックジャック(nbj)の場合 => ディーラーもnbjなら引き分け，それ以外は勝ち
        elif self.player_hand.is_nbj():
            if self.dealer_hand.is_nbj():
                return 'draw', 1.0
            else:
                return 'win', 2.5

        # それ以外の場合
        else:

            # => ディーラーがバーストしているなら勝ち
            if self.dealer_hand.is_busted():
                return 'win', 2.0

            # => ディーラーがnbjなら負け
            elif self.dealer_hand.is_nbj():
                return 'lose', 0.0

            # => それ以外ならスコア比較
            else:
                player_score = self.player_hand.get_score()
                dealer_score = self.dealer_hand.get_score()
                if player_score > dealer_score:
                    return 'win', 2.0
                elif player_score < dealer_score:
                    return 'lose', 0.0
                else:
                    return 'draw', 1.0

    # ゲーム開始時の処理
    # カードシャッフルを行った場合は True を，そうでない場合は False を返す
    def initialize_game(self):

        return_value = False

        # 必要ならカードシャッフル
        if self.game_ID % self.shuffle_interval == 0:
            self.card_set.shuffle()
            return_value = True

        # ゲームIDを 1 増やす
        self.game_ID += 1

        # 前ゲームの手札をクリア
        self.dealer_hand.clear()
        self.player_hand.clear()

        # ディーラーとプレイヤーに2枚ずつカードを配る
        self.dealer_hand.append(self.card_set.draw()) # ディーラーの1枚目
        self.dealer_hand.append(self.card_set.draw()) # ディーラーの2枚目
        self.player_hand.append(self.card_set.draw()) # プレイヤーの1枚目
        self.player_hand.append(self.card_set.draw()) # プレイヤーの2枚目

        return return_value

    # カードの残り枚数を取得
    def get_num_remaining_cards(self):
        return self.card_set.remaining_cards()

    # プレイヤーがバーストしたか否かを取得
    def player_is_busted(self):
        return self.player_hand.is_busted()

    # プレイヤーに1枚カードを配布
    def draw_player_card(self):
        self.player_hand.append(self.card_set.draw())

    # ルールに従ってディーラーカードを追加
    # （ディーラーのスコアが17以上になるか，カード使用枚数の上限に達するまで追加）
    def draw_dealer_cards(self):
        while self.dealer_hand.get_score() < 17 and len(self.dealer_hand.cards) < self.max_cards_per_game:
            self.dealer_hand.append(self.card_set.draw())

    # プレイヤーに初期カード情報を送信
    #   - psoc: プレイヤーとの間でのメッセージを送受信するためのソケット
    def send_init_cards(self, psoc: socket.socket):

        # ディーラーカード，プレイヤーカード1枚目，プレイヤーカード2枚目の順で送信
        dc = Dealer.get_info(self.dealer_hand.cards[0])
        pc1 = Dealer.get_info(self.player_hand.cards[0])
        pc2 = Dealer.get_info(self.player_hand.cards[1])
        psoc.send(bytes("{0},{1},{2}".format(dc, pc1, pc2), 'utf-8'))

    # プレイヤーへのメッセージ送信
    #   - psoc: プレイヤーとの間でのメッセージを送受信するためのソケット
    #   - rate: 配当倍率
    #   - status: プレイヤーの現在ステータス（勝敗結果，バーストしたか否か，サレンダーを受け付けたか否か，などを表す文字列）
    #   - send_player_card: プレイヤーカードの種類を通知するか否か
    #   - send_dealer_cards: ディーラーの手札を通知するか否か
    def send_message(self, psoc: socket.socket, rate: float, status: str, send_player_card=False, send_dealer_cards=False):
        if send_player_card:
            msg = '{0},'.format(Dealer.get_info(self.player_hand.cards[-1]))
        else:
            msg = ''
        score = self.player_hand.get_score()
        msg += '{0},{1},{2}'.format(score, status, rate)
        if send_dealer_cards:
            for i in range(1, len(self.dealer_hand.cards)):
                msg += ',{0}'.format(Dealer.get_info(self.dealer_hand.cards[i]))
        psoc.send(bytes(msg, 'utf-8'))

    # プレイヤーからのメッセージ受信
    # プレイヤーが選択した行動の種類を戻り値として返す
    #   - psoc: プレイヤーとの間でのメッセージを送受信するためのソケット
    def receive_message(self, psoc: socket.socket):
        msg = psoc.recv(1024).decode("utf-8")
        if msg == 'hit':
            return Action.HIT
        elif msg == 'stand':
            return Action.STAND
        elif msg == 'double_down':
            return Action.DOUBLE_DOWN
        elif msg == 'surrender':
            return Action.SURRENDER
        else:
            return Action.UNDEFINED


### ここから処理開始 ###

# ディーラークラスのインスタンスを作成
dealer = Dealer(n_decks=N_DECKS, shuffle_interval=SHUFFLE_INTERVAL, max_cards_per_game=MAX_CARDS_PER_GAME)

# プレイヤーからの接続を受け付けるソケットを用意
soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # プレイヤーからの通信受付用ソケット
soc.settimeout(1.0)
soc.bind((socket.gethostname(), PORT))
soc.listen(1)
print('The dealer program has started!!')
print()
print('Wainting for a new player ...')

# Ctrl+C で停止されるまで，無限ループでゲームを続ける
while True:

    try:
        player_soc, address = soc.accept() # プレイヤーからの通信待ち状態に入る
    except socket.timeout:
        pass
    except:
        soc.close()
        raise
    else:

        print('A player has come.')

        # 手札を初期化
        cardset_shuffled = dealer.initialize_game()
        if cardset_shuffled is True:
            print('Card set has been shuffled.') # 初期化中にカードをシャッフルした場合はメッセージを表示

        print('Num. remaining cards: ', dealer.get_num_remaining_cards())
        print('Game start!!')

        # ディーラーカード1枚とプレイヤーカード2枚をプレイヤーに開示
        dealer.send_init_cards(player_soc)

        # プレイヤーのアクションを受信して応答する（ループ処理）
        while True:

            action = dealer.receive_message(player_soc)

            # HIT の場合
            if action == Action.HIT:
                print("The player's action: HIT")

                # プレイヤーにカードを1枚配布
                dealer.draw_player_card()

                if dealer.player_is_busted():
                    # プレイヤーがバーストした場合
                    status = 'bust'
                    dealer.send_message(psoc=player_soc, rate=0.0, status=status, send_player_card=True, send_dealer_cards=True)
                else:
                    # プレイヤーがバーストしなかった場合
                    status = 'unsettled'
                    dealer.send_message(psoc=player_soc, rate=0.0, status=status, send_player_card=True)

            # STAND の場合
            elif action == Action.STAND:
                print("The player's action: STAND")

                # ルールに従ってディーラーにカードを追加
                dealer.draw_dealer_cards()

                # 勝敗を判定
                status, rate = dealer.judge()
                dealer.send_message(psoc=player_soc, rate=rate, status=status, send_dealer_cards=True)

            # DOUBLE DOWNの場合
            elif action == Action.DOUBLE_DOWN:
                print("The player's action: DOUBLE_DOWN")

                # プレイヤーにカードを1枚配布
                dealer.draw_player_card()

                if dealer.player_is_busted():
                    # プレイヤーがバーストした場合
                    status = 'bust'
                    rate = 0.0
                else:
                    # プレイヤーがバーストしなかった場合 => STAND の場合と同じ処理を実行
                    dealer.draw_dealer_cards()
                    status, rate = dealer.judge()
                dealer.send_message(psoc=player_soc, rate=rate, status=status, send_player_card=True, send_dealer_cards=True)

            # SURRENDERの場合
            elif action == Action.SURRENDER:
                print("The player's action: SURRENDER")
                status = 'surrendered'
                dealer.send_message(psoc=player_soc, rate=0.5, status=status, send_dealer_cards=True)

            # 定義されていないアクションは終了要求とみなす
            else:
                status = 'finished'

            print("The player's status: ", status)
            if status != 'unsettled':
                break # HITしてバーストしなかった場合を除き，ゲーム終了

        # 通信終了
        player_soc.close()
        print('The game has finished!')
        print()
        print('Wainting for a new player ...')
