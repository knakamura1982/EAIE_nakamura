import sys
import time
import argparse
import numpy as np
import tkinter as tk
from classes import Strategy, QTable
from PIL import Image, ImageTk, ImageDraw, ImageFont
from enum import Enum


# ----- ゲーム設定（ここから）-----
WIDTH = 12 # 盤面の横サイズ（セル数）
HEIGHT = 5 # 盤面の縦サイズ（セル数）
N_ENEMIES = 4 # 敵の数
N_TREASURES = 3 # 宝の数（初期値）
TIMESTEP = 0.4 # ゲーム内での 1 タイムステップが現実の何秒に相当するか
# ----- ゲーム設定（ここまで）-----


# ミニゲームを実行するクラス
class GameField:

    # ----- スコア設定（ここから）-----
    # 以下は強化学習における報酬としても利用されます．
    GOAL_SCORE = 50 # ゴールに到達した場合は +50 点
    ENEMY_SCORE = -100 # 敵に接触した場合は -100点
    TREASURE_SCORE = 10 # 宝を取得した場合は +10 点
    WALL_SCORE = -1 # 壁にぶつかった場合は -1 点（これは報酬としてのみ使用）
    # ----- スコア設定（ここまで）-----

    # ----- 強化学習の設定（ここから）-----
    LEARNING_RATE = 0.1 # 学習率
    DISCOUNT_FACTOR = 0.9 # 割引率
    EPSILON = 0.1 # ε-greedyにおけるε
    # ----- 強化学習の設定（ここから）-----

    # 自機がとる行動の種類
    class Action(Enum):
        UNDEFINED = 0 # 未定義値
        PROCEED = 1 # 右へ進む
        DOWN = 2 # 下に移動する
        UP = 3 # 上に移動する
        STAY = 4 # その場に留まる

    # 行動名の取得
    @staticmethod
    def get_action_name(action: Action):
        if action == GameField.Action.PROCEED:
            return 'PROCEED'
        elif action == GameField.Action.DOWN:
            return 'DOWN'
        elif action == GameField.Action.UP:
            return 'UP'
        elif action == GameField.Action.STAY:
            return 'STAY'
        else:
            return 'UNDEFINED'

    # 盤面画像におけるセルのサイズ（正方形, 定数, 単位は pixel ）
    CELL_SIZE = 48

    # 描画時に 1 タイムステップを更に何分割するか
    N_SUBSTEPS = 12

    # コンストラクタ
    #   - width: 盤面における1行あたりのセル数
    #   - height: 盤面における1列あたりのセル数
    #   - n_enemies: 配置する敵の数
    #   - n_treasures: 配置する宝の数
    def __init__(self, width, height, n_enemies, n_treasures):
        self.width = width
        self.height = height
        self.n_enemies = n_enemies
        self.n_treasures = n_treasures
        self.score = 0
        self.q_table = QTable(action_class=GameField.Action)
        self.g_keycode = -1

    # 盤面画像に自機/敵/宝マークを1つ描画する
    #   - draw: 描画先の画像
    #   - pos: 描画対象のセルの位置
    #   - ppos: 1時刻前に当該の自機/敵/宝が存在していたセル
    #   - char: マーク内に記入する文字（自機:'P', 敵:'E', 宝:'T'）
    #   - mcolor: マークの色
    #   - ccolor: マーク内の文字の色
    def __draw_mark(self, draw, pos, ppos, char, mcolor, ccolor, substep):
        if char == 'T':
            ltpos = ppos * self.CELL_SIZE
        else:
            if np.all(pos == ppos):
                ltpos = pos * self.CELL_SIZE + np.asarray([substep % 2, 0] if char == 'P' else [0, substep % 2])
            else:
                ltpos = ((substep * pos + (self.N_SUBSTEPS - substep) * ppos) / self.N_SUBSTEPS) * self.CELL_SIZE
            ltpos = ltpos.astype(np.int32)
        t = (ltpos[0] + self.CELL_SIZE // 3, ltpos[1] + self.CELL_SIZE // 5)
        c = (ltpos[0] + self.CELL_SIZE // 2, ltpos[1] + self.CELL_SIZE // 2)
        r = self.CELL_SIZE // 3
        draw.ellipse([(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)], fill=mcolor, width=2)
        draw.text(t, char, fill=ccolor, align='center', font=ImageFont.truetype("arial.ttf", 26))

    # ゴールを示す「G」マークを描画する
    #   - draw: 描画先の画像
    def __draw_goal(self, draw):
        ltpos = self.CELL_SIZE * self.goal_pos
        rbpos = self.CELL_SIZE * (self.goal_pos + np.asarray([1, 1]))
        t = (ltpos[0] + self.CELL_SIZE // 4, ltpos[1] + self.CELL_SIZE // 8)
        draw.rectangle([tuple(ltpos), tuple(rbpos)], fill=(0, 255, 0), outline='black') # ゴールしたセルを緑で塗り潰す
        draw.text(t, 'G', fill='black', align='center', font=ImageFont.truetype('arial.ttf', 32))

    # 盤面画像に自機／敵／宝マークを全て描画する
    #   - draw: 描画先の画像
    def __draw_units(self, draw, substep):
        self.__draw_goal(draw)
        for i in range(self.n_enemies):
            self.__draw_mark(draw, self.enemy_pos[i], self.prev_enemy_pos[i], 'E', (0, 0, 0), (255, 255, 255), substep) # 敵：黒マークに白字
        for i in range(self.n_treasures):
            if self.prev_treasure_pos[i, 0] >= 0: # 未取得の宝のみ描画
                self.__draw_mark(draw, self.treasure_pos[i], self.prev_treasure_pos[i], 'T', (128, 128, 0), (255, 255, 255), substep) # 宝：黄色マークに白字
        self.__draw_mark(draw, self.my_pos, self.prev_my_pos, 'P', (0, 0, 255), (255, 255, 255), substep) # 自機：青マークに白字

    # 敵に接触したことを示す「X」マークを描画する
    #   - draw: 描画先の画像
    def __draw_crush(self, draw):
        if self.crush_pos is not None: # 敵と接触した時のみ以下を実行
            ltpos = tuple(self.crush_pos * self.CELL_SIZE)
            lbpos = tuple((self.crush_pos + np.asarray([0, 1])) * self.CELL_SIZE)
            rtpos = tuple((self.crush_pos + np.asarray([1, 0])) * self.CELL_SIZE)
            rbpos = tuple((self.crush_pos + np.asarray([1, 1])) * self.CELL_SIZE)
            draw.rectangle([tuple(ltpos), tuple(rbpos)], fill=(255, 0, 0), outline='black') # 接触が発生したセルを赤で塗り潰す
            draw.line([ltpos, rbpos], fill='black', width=2)
            draw.line([lbpos, rtpos], fill='black', width=2)

    # 現状態を示す状態コードを求める
    def __get_state_code(self):
        s = np.zeros((self.height, 3), dtype=np.uint8)
        if self.my_pos[0] >= self.width - 2:
            s[ : , self.width-self.my_pos[0]: ] = 1
        s[self.my_pos[1], 0] = 2
        for i in range(self.n_enemies):
            x = self.enemy_pos[i, 0] - self.my_pos[0]
            if 0 <= x and x <= 2:
                s[self.enemy_pos[i, 1], x] = 3
        for i in range(self.n_treasures):
            x = self.treasure_pos[i, 0] - self.my_pos[0]
            if 0 <= x and x <= 2:
                s[self.treasure_pos[i, 1], x] = 4
        x = self.goal_pos[0] - self.my_pos[0]
        if 0 <= x and x <= 2:
            s[self.goal_pos[1], x] = 5
        s = s.reshape(self.height * 3)
        code = ''
        for i in range(len(s)):
            code += str(s[i])
        return code

    # キーボード操作を処理する関数
    def key_handler(self, e):
        self.g_keycode = e.keycode

    # 行動選択関数
    def __select_action(self, state, strategy: Strategy):
        g_keycode = self.g_keycode
        if strategy == Strategy.MANUAL:
            if g_keycode == 38: # 上キー（ubuntuでは'a'キーかも）
                a = self.Action.UP
            elif g_keycode == 39: # 右キー（ubuntuでは's'キーかも）
                a = self.Action.PROCEED
            elif g_keycode == 40: # 下キー（ubuntuでは'd'キーかも）
                a = self.Action.DOWN
            else:
                a = self.Action.STAY
            self.g_keycode = -1
            return a
        elif strategy == Strategy.QMAX:
            return self.q_table.get_best_action(state)
        elif strategy == Strategy.E_GREEDY:
            p = np.random.rand()
            if p < self.EPSILON:
                return self.__select_action(state, strategy=Strategy.RANDOM)
            else:
                return self.q_table.get_best_action(state)
        else:
            return np.random.choice([GameField.Action.PROCEED, GameField.Action.DOWN, GameField.Action.UP, GameField.Action.STAY])

    # 現在の盤面の状態をウィンドウに描画する
    def draw(self, substep=0):
        img = make_grid_image(self.width, self.height, self.CELL_SIZE)
        pil_img = Image.fromarray(img)
        pil_draw = ImageDraw.Draw(pil_img)
        self.__draw_units(pil_draw, substep)
        if substep == 0 or substep >= self.N_SUBSTEPS:
            self.__draw_crush(pil_draw)
        img = np.array(pil_img)
        return img

    # 現在の盤面状況を出力
    def print_status(self):
        print('player pos: x={0}, y={1}'.format(self.my_pos[0], self.my_pos[1]), file=sys.stderr)
        for i in range(self.n_enemies):
            print('enemy {0} pos: x={1}, y={2}'.format(i+1, self.enemy_pos[i, 0], self.enemy_pos[i, 1]), file=sys.stderr)
        for i in range(self.n_treasures):
            if self.treasure_pos[i, 0] >= 0:
                print('treasure {0} pos: x={1}, y={2}'.format(i+1, self.treasure_pos[i, 0], self.treasure_pos[i, 1]), file=sys.stderr)
        if self.crush_pos is not None:
            print('GAME OVER: player has collided with the enemy {0}! score {1}!!'.format(self.crush_id+1, self.ENEMY_SCORE), file=sys.stderr)
            print('CURRENT SCORE: {0}'.format(self.score), file=sys.stderr)
        if np.all(self.goal_pos == self.my_pos):
            print('GAME CLEAR: player has reached the goal! score +{0}!!'.format(self.GOAL_SCORE), file=sys.stderr)
            print('CURRENT TOTAL SCORE: {0}'.format(self.score), file=sys.stderr)

    # 行動を実行
    def __act(self, a):

        reward = 0 # 即時報酬
        over_flag = False # ゲーム終了フラグ（Falseで初期化）

        if a == self.Action.PROCEED:
            print('player action: PROCEED', file=sys.stderr)
            if self.my_pos[0] == self.width - 1:
                reward += self.WALL_SCORE # 壁にぶつかった場合は即時報酬を -1 する
            self.my_pos[0] = min(self.my_pos[0] + 1, self.width - 1)
        elif a == self.Action.DOWN:
            print('player action: DOWN', file=sys.stderr)
            if self.my_pos[1] == self.height - 1:
                reward += self.WALL_SCORE # 壁にぶつかった場合は即時報酬を -1 する
            self.my_pos[1] = min(self.my_pos[1] + 1, self.height - 1)
        elif a == self.Action.UP:
            print('player action: UP', file=sys.stderr)
            if self.my_pos[1] == 0:
                reward += self.WALL_SCORE # 壁にぶつかった場合は即時報酬を -1 する
            self.my_pos[1] = max(self.my_pos[1] - 1, 0)
        else:
            print('player action: STAY', file=sys.stderr)

        # 敵を移動させる
        for i in range(self.n_enemies):
            if self.enemy_pos[i, 1] == 0:
                v = np.random.randint(1, 3)
            elif self.enemy_pos[i, 1] == self.height - 1:
                v = np.random.randint(2)
            else:
                v = np.random.randint(3)
            self.enemy_pos[i, 1] = max(0, min(self.height - 1, self.enemy_pos[i, 1] + (v-1)))
            if np.all(self.enemy_pos[i] == self.my_pos) or (np.all(self.enemy_pos[i] == self.prev_my_pos) and np.all(self.prev_enemy_pos[i] == self.my_pos)):
                # 敵と接触した場合
                reward += self.ENEMY_SCORE # 即時報酬を -100 する
                self.score += self.ENEMY_SCORE
                self.crush_pos = self.my_pos
                self.crush_id = i
                self.finished = True
                over_flag = True

        # ゴール判定
        if np.all(self.goal_pos == self.my_pos):
            reward += self.GOAL_SCORE # 即時報酬を +50 する
            self.score += self.GOAL_SCORE
            self.finished = True
            over_flag = True

        self.print_status()

        # 宝との接触を判定
        for i in range(self.n_treasures):
            if np.all(self.treasure_pos[i] == self.my_pos): # 宝と接触した場合
                reward += self.TREASURE_SCORE # 即時報酬を +10 する
                self.score += self.TREASURE_SCORE
                self.treasure_pos[i, 0] = self.treasure_pos[i, 1] = -1 # 取得済みを表す値をセット
                print('get treasure {0}! score +{1}!!'.format(i+1, self.TREASURE_SCORE), file=sys.stderr)

        print('', file=sys.stderr)

        return reward, over_flag

    # ゲーム開始状態にする
    def start(self):

        # 時刻を 0 に初期化
        self.timestep = 0
        self.finished = False

        # 敵接触判定を初期化
        self.crush_id = -1
        self.crush_pos = None

        # 自機の初期位置を設定する
        self.my_pos = np.asarray((0, np.random.randint(0, self.height)))
        self.prev_my_pos = self.my_pos.copy()

        # ゴール位置を設定する
        self.goal_pos = np.asarray((self.width - 1, np.random.randint(0, self.height)))

        # 敵を配置する（敵の初期位置を設定する）
        a = np.random.permutation(np.arange(1, self.width - 1))
        self.enemy_pos = []
        for i in range(self.n_enemies):
            self.enemy_pos.append((a[i], np.random.randint(0, self.height)))
        self.enemy_pos = np.asarray(self.enemy_pos)
        self.prev_enemy_pos = self.enemy_pos.copy()

        # 宝を配置する（宝の初期位置を設定する）
        self.treasure_pos = []
        for i in range(self.n_treasures):
            self.treasure_pos.append((a[i + self.n_enemies], np.random.randint(0, self.height)))
        self.treasure_pos = np.asarray(self.treasure_pos)
        self.prev_treasure_pos = self.treasure_pos.copy()

        self.print_status()
        print('', file=sys.stderr)

    # 1時刻分だけゲームを進める
    # その結果ゲームが終了した場合は True を，そうでない場合は False を返す
    def step(self, learning_mode=False, manual_mode=False):

        if self.finished:
            return True

        self.prev_my_pos = self.my_pos.copy()
        self.prev_enemy_pos = self.enemy_pos.copy()
        self.prev_treasure_pos = self.treasure_pos.copy()

        # 現状態を取得
        state = self.__get_state_code()

        # 時刻変数を1増やす
        self.timestep += 1

        # 自機の行動を選択
        strategy = Strategy.MANUAL if manual_mode else (Strategy.E_GREEDY if learning_mode else Strategy.QMAX)
        action = self.__select_action(state, strategy)

        # 選択した行動を実際に実行
        # 戻り値:
        #   - done: 終了フラグ．今回の行動によりゲームが終了したか否か（終了した場合はTrue, 続行中ならFalse）
        #   - reward: 即時報酬
        reward, done = self.__act(action)

        if learning_mode:

            # 「現在の状態」を再取得
            prev_state = state # 行動前の状態を別変数に退避
            state = self.__get_state_code()

            # Qテーブルを更新
            _, V = self.q_table.get_best_action(state, with_value=True)
            Q = self.q_table.get_Q_value(prev_state, action) # 現在のQ値
            Q = (1 - self.LEARNING_RATE) * Q + self.LEARNING_RATE * (reward + self.DISCOUNT_FACTOR * V) # 新しいQ値
            self.q_table.set_Q_value(prev_state, action, Q) # 新しいQ値を登録

        return done


# 盤面を表現するグリッド画像を作成する
#   - width: 画像の横幅(pixel)
#   - height: 画像の縦幅(pixel)
#   - cellsize: グリッド線同士の間隔(pixel)
def make_grid_image(width, height, cellsize):

    # まず，白一色の画像を作る
    img = 255 * np.ones((height * cellsize, width * cellsize, 3), dtype=np.uint8)

    # 横線（黒）を引く
    for i in range(1, height):
        img[ cellsize * i , : ] = np.asarray([0, 0, 0])

    # 縦線（黒）を引く
    for i in range(1, width):
        img[ : , cellsize * i ] = np.asarray([0, 0, 0])

    # 結果を返却
    return img


# 盤面を描画するウィンドウを表すクラス
class Window:

    CANVAS_SIZE = [ WIDTH * GameField.CELL_SIZE , HEIGHT * GameField.CELL_SIZE ] # カンバスのサイズ
    WINDOW_SIZE = [ CANVAS_SIZE[0] + 20 , CANVAS_SIZE[1] + 60 ] # ウィンドウのサイズ
    WINDOW_TITLE = 'Mini Game'

    def __init__(self):
        img = make_grid_image(WIDTH, HEIGHT, GameField.CELL_SIZE)
        self.root = tk.Tk()
        self.root.title(self.WINDOW_TITLE)
        self.root.geometry('{0}x{1}'.format(self.WINDOW_SIZE[0], self.WINDOW_SIZE[1]))
        self.button = tk.Button(width=12, text='auto play', font=('Arial', '10', 'bold'), command=self.run)
        self.play_button = tk.Button(width=12, text='manual play', font=('Arial', '10', 'bold'), command=self.play)
        self.learn_button = tk.Button(width=12, text='learn', font=('Arial', '10', 'bold'), command=self.learn)
        self.canvas = tk.Canvas(width=self.CANVAS_SIZE[0], height=self.CANVAS_SIZE[1])
        self.textbox = tk.Entry(width=6, font=('Arial', '10', 'bold'))
        self.button.place(x=140, y=10)
        self.play_button.place(x=10, y=10)
        self.learn_button.place(x=270, y=10)
        self.canvas.place(x=10, y=45)
        self.textbox.place(x=390, y=15)
        self.canvas.photo = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas_img = self.canvas.create_image(0, 0, image=self.canvas.photo, anchor=tk.NW)
        self.field = GameField(WIDTH, HEIGHT, N_ENEMIES, N_TREASURES)
        self.root.bind("<KeyPress>", self.field.key_handler)
        self.history_file = None

    def draw(self, substep):
        img = self.field.draw(substep)
        self.canvas.photo = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas.itemconfig(self.canvas_img, image=self.canvas.photo)
        self.root.update()

    def run(self):
        self.button['state'] = tk.DISABLED
        self.play_button['state'] = tk.DISABLED
        self.learn_button['state'] = tk.DISABLED
        self.textbox['state'] = tk.DISABLED
        self.field.start()
        while True:
            for i in range(GameField.N_SUBSTEPS):
                self.draw(substep=i+1)
                time.sleep(TIMESTEP / GameField.N_SUBSTEPS)
            if self.field.step() == True:
                break
        for i in range(GameField.N_SUBSTEPS):
            self.draw(substep=i+1)
            time.sleep(TIMESTEP / GameField.N_SUBSTEPS)
        self.button['state'] = tk.NORMAL
        self.play_button['state'] = tk.NORMAL
        self.learn_button['state'] = tk.NORMAL
        self.textbox['state'] = tk.NORMAL
        self.root.update()

    def play(self):
        self.button['state'] = tk.DISABLED
        self.play_button['state'] = tk.DISABLED
        self.learn_button['state'] = tk.DISABLED
        self.textbox['state'] = tk.DISABLED
        self.field.start()
        while True:
            for i in range(GameField.N_SUBSTEPS):
                self.draw(substep=i+1)
                time.sleep(TIMESTEP / GameField.N_SUBSTEPS)
            if self.field.step(manual_mode=True) == True:
                break
        for i in range(GameField.N_SUBSTEPS):
            self.draw(substep=i+1)
            time.sleep(TIMESTEP / GameField.N_SUBSTEPS)
        self.button['state'] = tk.NORMAL
        self.play_button['state'] = tk.NORMAL
        self.learn_button['state'] = tk.NORMAL
        self.textbox['state'] = tk.NORMAL
        self.root.update()

    def learn(self):
        self.button['state'] = tk.DISABLED
        self.play_button['state'] = tk.DISABLED
        self.learn_button['state'] = tk.DISABLED
        self.textbox['state'] = tk.DISABLED
        self.root.update()
        if self.history_file is not None:
            hf = open(self.history_file, 'w')
        iter = int(self.textbox.get())
        for i in range(iter):
            self.field.start()
            while True:
                if self.field.step(learning_mode=True) == True:
                    break
            if self.history_file is not None:
                print(self.field.score, file=hf)
        if self.history_file is not None:
            hf.close()
        self.button['state'] = tk.NORMAL
        self.play_button['state'] = tk.NORMAL
        self.learn_button['state'] = tk.NORMAL
        self.textbox['state'] = tk.NORMAL
        self.root.update()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Q Learning Demo')
    parser.add_argument('--load', type=str, default='', help='filename of Q table to be loaded before learning')
    parser.add_argument('--save', type=str, default='', help='filename where Q table will be saved after learning')
    parser.add_argument('--history', type=str, default='mygame_score_log.csv', help='filename where history of scores will be saved during learning')
    args = parser.parse_args()

    # ゲームウィンドウのインスタンスを作成
    win = Window()

    # Q table をロード
    if args.load != '':
        win.field.q_table.load(args.load)

    # スコア履歴保存ファイルを設定
    if args.history != '':
        win.history_file = args.history

    # ゲームウィンドウを表示
    win.root.mainloop()

    # Q table をセーブ
    if args.save != '':
        win.field.q_table.save(args.save)
