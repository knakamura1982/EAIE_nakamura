### 全体を通して

主にOpenAI Gymの制御系タスクを題材としてQ学習を試みるプログラム群です．  
これらのタスクは，本来，deep neural network でQ学習を実現する Deep Q Network (DQN) で解くことを想定されていますが，  
ここでは伝統的な（通常の）Q学習で解くことを試みます（このため，あまり上手くはいかないかもしれません）．

取り上げるのは，最も単純と思われる以下の4つのタスクです．  
下記のサイトに各タスクの情報が簡単にまとめられていますので，併せてご参照ください．  
https://github.com/openai/gym/wiki/Leaderboard

### CartPole_v0.py

OpenAI Gym の Classic Control タスクの一つ CartPole-v0 を通常のQ学習で解くプログラム（のひな型）．  
タスクの内容については下記サイトなどを参照．  
https://github.com/openai/gym/wiki/CartPole-v0

**コマンド**
```
# 一から学習する場合
python CartPole_v0.py --games 10 --max_steps 200 --save result_QTable.pkl

# 学習済みQテーブルをロードし，そこから学習を再開する場合
python CartPole_v0.py --games 10 --max_steps 200 --load initial_QTable.pkl --save result_QTable.pkl

# 学習済みQテーブルをロードして単にタスクを実行するだけの場合
python CartPole_v0.py --games 10 --max_steps 200 --load initial_QTable.pkl --testmode

# Qテーブルを用いずにランダム戦略でタスクを実行する場合
python CartPole_v0.py --games 10 --max_steps 200 --randmode
```
**オプション**
- games
　- 連続して何ゲームプレイするか
　- 指定しない場合，デフォルト値として 1 がセットされる
- max_steps
  - 1ゲームあたりの最大ステップ数
  - ステップ数がこのオプションで指定した値を超えると，終了条件を満たしていなくても現在のゲームを打ち切り，次のゲームに移行する
  - 指定しない場合，デフォルト値として 200 がセットされる
- load
  - 指定したファイルからQテーブルをロードする
  - 必ずしも指定しなくても良い．指定しない場合，Qテーブルは全てのフィールドが 0 で初期化される
- save
  - 指定したファイルに学習結果のQテーブルが保存される（既に存在するファイル名を指定した場合は上書きされる）
  - 必ずしも指定しなくても良い．指定しない場合，学習結果は保存されない（プログラム終了時に破棄される）
- testmode
  - 指定すると，常にQ値最大の行動を選択するようになる（ε-greedy において ε=0 の状態になる）
  - このモードで動作しているときはQテーブルは更新されない
- randmode
  - 指定すると，Qテーブルを用いずにランダム戦略でタスクを実行する
  - このモードで動作しているときはQテーブルは更新されない
  - testmode より優先される（両方指定した場合はランダム戦略になる）

### MountainCar_v0.py

OpenAI Gym の Classic Control タスクの一つ MountainCar-v0 を通常のQ学習で解くプログラム（のひな型）．  
タスクの内容については下記サイトなどを参照．  
https://github.com/openai/gym/wiki/MountainCar-v0

### Acrobot_v1.py

OpenAI Gym の Classic Control タスクの一つ Acrobot-v1 を通常のQ学習で解くプログラム（のひな型）．  
タスクの内容については下記サイトなどを参照．  
https://www.tcom242242.net/entry/ai-2/%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92/acrobot/

### LunarLander_v2.py

OpenAI Gym の Box2D タスクの一つ LunarLander-v2 を通常のQ学習で解くプログラム（のひな型）．  
タスクの内容については下記サイトなどを参照．  
https://shiva-verma.medium.com/solving-lunar-lander-openaigym-reinforcement-learning-785675066197
