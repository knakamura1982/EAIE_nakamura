## 全体を通して

主に OpenAI Gym (gymnasium) の制御系タスクを題材としてQ学習を試みるプログラム群です．  
これらのタスクは，本来，deep neural network でQ学習を実現する Deep Q Network (DQN) で解くことを想定されていますが，  
ここでは伝統的な（通常の）Q学習で解くことを試みます（このため，あまり上手くはいかないかもしれません）．

取り上げるのは，最も単純と思われる以下の4つのタスクです．  
下記のサイトに各タスクの情報が簡単にまとめられていますので，併せてご参照ください．  
https://gymnasium.farama.org/

## CartPole_v1.py

OpenAI Gym (gymnasium) の Classic Control タスクの一つ CartPole-v1 を通常のQ学習で解くプログラム（のひな型）．  
タスクの内容については下記サイトなどをご参照ください．  
https://gymnasium.farama.org/environments/classic_control/cart_pole/

**コマンド例**
```
# 一から学習する場合
python CartPole_v1.py --games 10 --max_steps 200 --save QTable.pkl

# 学習済みQテーブルをロードし，そこから学習を再開して新たなQテーブルを作成・保存する場合
python CartPole_v1.py --games 10 --max_steps 200 --load QTable.pkl --save new_QTable.pkl

# 学習済みQテーブルをロードして単にタスクを実行するだけの場合
python CartPole_v1.py --games 10 --max_steps 200 --load QTable.pkl --testmode

# Qテーブルを用いずにランダム戦略でタスクを実行する場合
python CartPole_v1.py --games 10 --max_steps 200 --randmode
```
**オプション**
- games
  - 連続して何ゲームプレイするか
  - 指定しない場合，デフォルト値として 1 がセットされます．
- max_steps
  - 1ゲームあたりの最大ステップ数
  - ステップ数がこのオプションで指定した値を超えると，  
  終了条件を満たしていなくても現在のゲームを打ち切り，次のゲームに移行します．  
  - 指定しない場合，デフォルト値として 200 がセットされます．
- load
  - 指定したファイルからQテーブルをロードします．
  - 必ずしも指定する必要はありません．指定しない場合，Qテーブルは全てのフィールドが 0 で初期化されます．
- save
  - 指定したファイルに学習結果のQテーブルが保存されます（既に存在するファイル名を指定した場合は上書きされます）．
  - 必ずしも指定する必要はありません．指定しない場合，学習結果は保存されません（プログラム終了時に破棄されます）．
- testmode
  - 指定すると，常にQ値最大の行動を選択するようになります（ε-greedy における ε=0 の状態）．
  - このモードで動作しているときはQテーブルは更新されません．
- randmode
  - 指定すると，Qテーブルを用いずにランダム戦略でタスクを実行するようになります．
  - このモードで動作しているときはQテーブルは更新されません．
  - testmode より優先されます（両方指定した場合はランダム戦略になります）．

## MountainCar_v0.py

OpenAI Gym (gymnasium) の Classic Control タスクの一つ MountainCar-v0 を通常のQ学習で解くプログラム（のひな型）．  
タスクの内容については下記サイトなどをご参照ください．  
https://gymnasium.farama.org/environments/classic_control/mountain_car/

**コマンド例**
```
# 一から学習する場合
python MountainCar_v0.py --games 10 --max_steps 200 --save QTable.pkl

# 学習済みQテーブルをロードし，そこから学習を再開して新たなQテーブルを作成・保存する場合
python MountainCar_v0.py --games 10 --max_steps 200 --load QTable.pkl --save new_QTable.pkl

# 学習済みQテーブルをロードして単にタスクを実行するだけの場合
python MountainCar_v0.py --games 10 --max_steps 200 --load QTable.pkl --testmode

# Qテーブルを用いずにランダム戦略でタスクを実行する場合
python MountainCar_v0.py --games 10 --max_steps 200 --randmode
```
**オプション**
- いずれも CartPole_v1.py の同名オプションと全く同じです．

## Acrobot_v1.py

OpenAI Gym (gymnasium) の Classic Control タスクの一つ Acrobot-v1 を通常のQ学習で解くプログラム（のひな型）．  
タスクの内容については下記サイトなどをご参照ください．  
https://gymnasium.farama.org/environments/classic_control/acrobot/

**コマンド例**
```
# 一から学習する場合
python Acrobot_v1.py --games 10 --max_steps 200 --save QTable.pkl

# 学習済みQテーブルをロードし，そこから学習を再開して新たなQテーブルを作成・保存する場合
python Acrobot_v1.py --games 10 --max_steps 200 --load QTable.pkl --save new_QTable.pkl

# 学習済みQテーブルをロードして単にタスクを実行するだけの場合
python Acrobot_v1.py --games 10 --max_steps 200 --load QTable.pkl --testmode

# Qテーブルを用いずにランダム戦略でタスクを実行する場合
python Acrobot_v1.py --games 10 --max_steps 200 --randmode
```
**オプション**
- いずれも CartPole_v1.py の同名オプションと全く同じです．

## LunarLander_v2.py

OpenAI Gym (gymnasium) の Box2D タスクの一つ LunarLander-v2 を通常のQ学習で解くプログラム（のひな型）．  
タスクの内容については下記サイトなどをご参照ください．  
https://gymnasium.farama.org/environments/box2d/lunar_lander/

**コマンド例**
```
# 一から学習する場合
python LunarLander_v2.py --games 10 --max_steps 200 --save QTable.pkl

# 学習済みQテーブルをロードし，そこから学習を再開して新たなQテーブルを作成・保存する場合
python LunarLander_v2.py --games 10 --max_steps 200 --load QTable.pkl --save new_QTable.pkl

# 学習済みQテーブルをロードして単にタスクを実行するだけの場合
python LunarLander_v2.py --games 10 --max_steps 200 --load QTable.pkl --testmode

# Qテーブルを用いずにランダム戦略でタスクを実行する場合
python LunarLander_v2.py --games 10 --max_steps 200 --randmode
```
**オプション**
- いずれも CartPole_v1.py の同名オプションと全く同じです．

## myGame.py

これは OpenAI Gym (gymnasium) のタスクではなく，Q学習によるAI実装のデモ用に本リポジトリの作者が自作したミニゲームです．  
人間がキーボード操作でプレイすることも一応可能です．

**ゲーム内容**

- ゲーム開始時，縦 H マス，横 W マスの盤面に自機（P），敵（E），宝（T），ゴール（G）を配置
  - 自機の横位置は常に左端，ゴールの横位置は常に右端
  - 敵と宝の横位置はランダム，ただし同じ列に複数が配置されることはない
  - 縦位置はランダム（自機，敵，宝，ゴールの何れも）
  - なお，盤面のサイズ（H, W）や敵・宝の数は 13～16 行目の定数値をいじることにより変更可能
- 自機は4種類の行動を取れる
  - 右に1マス移動，上に1マス移動，下に1マス移動，その場に止まる
- 敵の行動は3種類
  - 上に1マス移動，下に1マス移動，その場に止まる
  - 敵は横移動はしない
- 宝とゴールの位置は不変
- 得点（Q学習における報酬としても使用）
  - 宝に触れると +10点
  - 敵に触れると -100点，ゲームオーバー
  - ゴールに到達すると +50 点，ゲームクリア
  - なお，これらの設定は 26～28 行目の定数値をいじることにより変更可能

**Q学習の設定**

- 自機の列とその前方2列分を環境状態とみなす
- 得点をそのまま報酬として活用
- 学習率: 0.1
- 割引率: 0.9
- ε-greedyにおけるε: 0.1
- なお，学習率，割引率，ε の値は 33～35 行目の定数値をいじることにより変更可能

**コマンド例**
```
python myGame.py --load QTable.pkl --save new_QTable.pkl --history mygame_score_log.csv
```
**オプション**
- load
  - プログラム開始時，指定したファイルからQテーブルをロードします．
  - 必ずしも指定する必要はありません．指定しない場合，Qテーブルは全てのフィールドが 0 で初期化されます．
- save
  - 指定したファイルに学習結果のQテーブルが保存されます（既に存在するファイル名を指定した場合は上書きされます）．
  - 必ずしも指定する必要はありません．指定しない場合，学習結果は保存されません（プログラム終了時に破棄されます）．
- history
  - 指定したファイルに毎ゲーム終了時点での得点が記録されます（既に存在するファイル名を指定した場合は上書きされます）．
  - 必ずしも指定する必要はありません．指定しない場合，得点は記録されません．

## QTable_checker.py

保存済みのQテーブルの内容を出力するプログラム．

**コマンド**
```
python QTable_checker.py --name myGame --file QTable.pkl

# 以下のようにして csv ファイルに書き出してから Excel で確認した方が分かりやすいかも
python QTable_checker.py --name myGame --file QTable.pkl > QTable.csv
```

**オプション**
- name
  - ゲームの名称（myGame, CartPole_v1, MountainCar_v0, Acrobot_v1, LunarLander_v2 のいずれか）．
- file
  - nameオプションで指定したタスクの実行時に保存されたQテーブルファイル．

**余談**

myGame.py のゲームで得られたQテーブルでは，AIによる観測状態（「現在の状態」）を文字列で表現しています．  
この文字列の意味は次の通りです．
```
例として，環境状態が縦5マス，横3マスで表される場合で考えます．

まず，各セルに以下のように番号を振ります（左上から右に向かって順に進み，右端に達したら次の列の左端へ）．
-------------
| a | b | c |
|---|---|---|
| d | e | f |
|---|---|---|
| g | h | i |
|---|---|---|
| j | k | l |
|---|---|---|
| m | n | o |
-------------

次に，各セルの状態を以下の数字で表現します．
  0: 何も存在しない
  1: 壁（自機が盤面の右端にいる場合は b,c 列のセルが，右端の一つ前にいる場合は c 列のセルが，それぞれこの状態になる）
  2: 自機が存在
  3: 敵が存在
  4: 宝が存在
  5: ゴール位置

その上で，数字を a,b,c,d,e,f,g,h,i,j,k,l,m,n,o の順で並べた文字列が，上述の「観測状態を表す文字列」です．

具体例を挙げます．
-------------
|   | T |   |
|---|---|---|
| P |   |   |
|---|---|---|
| E |   |   |
|---|---|---|
|   |   | G |
|---|---|---|
|   |   |   |
-------------
であったとします（P:自機, E:敵, T:宝, G:ゴール）．
上の状態を数値表現に直すと
-------------
| 0 | 4 | 0 |
|---|---|---|
| 2 | 0 | 0 |
|---|---|---|
| 3 | 0 | 0 |
|---|---|---|
| 0 | 0 | 5 |
|---|---|---|
| 0 | 0 | 0 |
-------------
であり，これを順に並べた 040200300005000 がこの状態を表す文字列となります．
この後，少し時間が進んで自機が右端に到達し，
-----
| P |
|---|
|   |
|---|
|   |
|---|
| G |
|---|
|   |
-----
になったとします．
このケースは，「壁」を表す「1」を用いて
-------------
| 2 | 1 | 1 |
|---|---|---|
| 0 | 1 | 1 |
|---|---|---|
| 0 | 1 | 1 |
|---|---|---|
| 5 | 1 | 1 |
|---|---|---|
| 0 | 1 | 1 |
-------------
と表現され，これを順に並べた 211011011511011 で現状態が表されます．
```

## qtable.py

Qテーブルの細かい処理を実装したファイル．
