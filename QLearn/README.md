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
