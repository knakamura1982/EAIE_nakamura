import os
import sys
sys.path.append(os.path.abspath('..'))
import torch
import torch.nn as nn
from mylib.basic_layers import *


# ニューラルネットワークのサンプル
# MLP (Multi-Layer Perceptron) と呼ばれるもの
class SampleMLP(nn.Module):

    def __init__(self):
        super(SampleMLP, self).__init__()

        ### 1層目 ###

        # 全結合層1: 2次元入力 → 10パーセプトロン
        self.fc1 = nn.Linear(in_features=2, out_features=10)
        # バッチ正規化層1
        self.bn1 = nn.BatchNorm1d(num_features=10) # 前の全結合層の出力パーセプトロン数が10なので
        # 活性化関数1: ReLU
        self.act1 = nn.ReLU()

        ### 2層目 ###

        # 全結合層2: 10パーセプトロン → 10パーセプトロン
        self.fc2 = nn.Linear(in_features=10, out_features=10)
        # ドロップアウト層2
        self.drop2 = nn.Dropout(p=0.5) # ドロップアウト率 0.5
        # 活性化関数2: Tanh
        self.act2 = nn.Tanh()

        ### 3層目 ###

        # 全結合層3: 10パーセプトロン → 3クラス出力
        self.fc3 = nn.Linear(in_features=10, out_features=3)

    def forward(self, x):
        h = self.fc1(x)   # 全結合層1にデータを入力
        h = self.bn1(h)   # 続いてバッチ正規化層1に通す
        h = self.act1(h)  # 続いて活性化関数1に通す
        h = self.fc2(h)   # 続いて全結合層2に通す
        h = self.drop2(h) # 続いてドロップアウト2に通す
        h = self.act2(h)  # 続いて活性化関数2に通す
        y = self.fc3(h)   # 最後に全結合層3に通す
        return y


# SampleMLP と同じニューラルネットワークを basic_layers.py に記載の自作モジュールで作成したもの
class myMLP(nn.Module):

    def __init__(self):
        super(myMLP, self).__init__()

        # 1層目: 2次元入力 → 10パーセプトロン（バッチ正規化あり, ドロップアウトなし, 活性化関数 ReLU）
        self.layer1 = FC(in_features=2, out_features=10, do_bn=True, activation='relu')

        # 2層目: 10パーセプトロン → 10パーセプトロン（バッチ正規化なし, ドロップアウトあり（ドロップアウト率 0.5）, 活性化関数 Tanh）
        self.layer2 = FC(in_features=10, out_features=10, do_bn=False, dropout_ratio=0.5, activation='tanh')

        # 3層目: 10パーセプトロン → 3クラス出力（バッチ正規化なし, ドロップアウトなし, 活性化関数なし）
        self.layer3 = FC(in_features=10, out_features=3, do_bn=False, activation='none')

    def forward(self, x):
        h = self.layer1(x) # 1層目にデータを入力
        h = self.layer2(h) # 続いて2層目に通す
        y = self.layer3(h) # 最後に3層目に通す
        return y


# CNN (Convolutional Neural Network) のサンプル
class SampleCNN(nn.Module):

    # C: 入力画像のチャンネル数（グレースケール画像なら1，カラー画像なら3）
    # H: 入力画像の縦幅（4の倍数を想定）
    # W: 入力画像の横幅（4の倍数を想定）
    # N: 認識対象のクラス数
    def __init__(self, C, H, W, N):
        super(SampleCNN, self).__init__()

        # 層ごとのチャンネル数
        L1_C = 8 # 1層目
        L2_C = 16 # 2層目
        L3_C = 32 # 3層目

        ### 1層目 ###

        # 畳込み層1: Cチャンネル入力 → L1_Cチャンネル，カーネルサイズ5x5
        # ストライド幅とゼロパディングを適切に設定する必要あり（ここでは，特徴マップの縦横幅が変わらないように設定）
        self.conv1 = nn.Conv2d(in_channels=C, out_channels=L1_C, kernel_size=5, stride=1, padding=2)
        # 活性化関数1: ELU
        self.act1 = nn.ELU()
        # プーリング層1: 平均値プーリング
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        ### 2層目 ###

        # 畳込み層2: L1_Cチャンネル → L2_Cチャンネル，カーネルサイズ3x3
        # ストライド幅とゼロパディングを適切に設定する必要あり（ここでは，特徴マップの縦横幅が変わらないように設定）
        self.conv2 = nn.Conv2d(in_channels=L1_C, out_channels=L2_C, kernel_size=3, stride=1, padding=1)
        # バッチ正規化層2
        self.bn2 = nn.BatchNorm2d(num_features=L2_C) # 前の畳込み層の出力チャンネル数がL2_Cなので
        # ドロップアウト層2
        self.drop2 = nn.Dropout(p=0.2) # ドロップアウト率 0.2
        # 活性化関数2: Leaky-ReLU
        self.act2 = nn.LeakyReLU()
        # プーリング層2: maxプーリング
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        ### 3層目 ###

        # 畳込み層3: L2_Cチャンネル → L3_Cチャンネル，カーネルサイズ3x3
        self.conv3 = nn.Conv2d(in_channels=L2_C, out_channels=L3_C, kernel_size=3, stride=1, padding=1)
        # バッチ正規化層3
        self.bn3 = nn.BatchNorm2d(num_features=L3_C) # 前の畳込み層の出力チャンネル数がL3_Cなので
        # 活性化関数3: ReLU
        self.act3 = nn.ReLU()
        # 平坦化層3
        self.flat3 = nn.Flatten()

        ### 4層目 ###

        # 全結合層4: L3_C*(H/4)*(W/4)個のパーセプトロン → 2N個のパーセプトロン
        self.fc4 = nn.Linear(in_features=L3_C*(H//4)*(W//4), out_features=2*N)
        # 活性化関数4: ReLU
        self.act4 = nn.ReLU()

        ### 5層目 ###

        # 全結合層5: 2N個のパーセプトロン → N個のパーセプトロン
        self.fc5 = nn.Linear(in_features=2*N, out_features=N)

    def forward(self, x):
        h = self.conv1(x) # 畳込み層1にデータを入力（この時点では特徴マップの縦幅・横幅は元のまま）
        h = self.act1(h)  # 続いて活性化関数1に通す
        h = self.pool1(h) # 続いてプーリング層1に通す（この時点で特徴マップの縦幅・横幅はそれぞれ元の 1/2 になる）
        h = self.conv2(h) # 続いて畳込み層2に通す（この時点では特徴マップの縦幅・横幅は元の 1/2 のまま）
        h = self.bn2(h)   # 続いてバッチ正規化層2に通す
        h = self.drop2(h) # 続いてドロップアウト層2に通す
        h = self.act2(h)  # 続いて活性化関数2に通す
        h = self.pool2(h) # 続いてプーリング層2に通す（この時点で特徴マップの縦幅・横幅はそれぞれ元の 1/4 になる）
        h = self.conv3(h) # 続いて畳込み層3に通す（この時点では特徴マップの縦幅・横幅は元の 1/4 のまま）
        h = self.bn3(h)   # 続いてバッチ正規化層3に通す
        h = self.act3(h)  # 続いて活性化関数3に通す
        h = self.flat3(h) # 続いて平坦化層3に通す（この時点で3次元的な構造を持つ特徴マップが一列に並べ直される）
        h = self.fc4(h)   # 続いて全結合層4に通す
        h = self.act4(h)  # 続いて活性化関数4に通す
        y = self.fc5(h)   # 最後に全結合層5に通す
        return y



# SampleCNN と同じニューラルネットワークを basic_layers.py に記載の自作モジュールで作成したもの
class myCNN(nn.Module):

    # C: 入力画像のチャンネル数（グレースケール画像なら1，カラー画像なら3）
    # H: 入力画像の縦幅（4の倍数を想定）
    # W: 入力画像の横幅（4の倍数を想定）
    # N: 認識対象のクラス数
    def __init__(self, C, H, W, N):
        super(myCNN, self).__init__()

        # 層ごとのチャンネル数
        L1_C = 8 # 1層目
        L2_C = 16 # 2層目
        L3_C = 32 # 3層目

        ### 1層目 ###

        # 畳込み層1: Cチャンネル入力 → L1_Cチャンネル特徴マップ（バッチ正規化なし, ドロップアウトなし, 活性化関数 ELU）
        # カーネルサイズ5x5，ストライド幅とゼロパディングは特徴マップの縦横幅が変わらないように自動設定
        self.conv1 = Conv(in_channels=C, out_channels=L1_C, kernel_size=5, do_bn=False, activation='elu')
        # プーリング層1: 平均値プーリング
        self.pool1 = Pool(method='avg')

        ### 2層目 ###

        # 畳込み層2: L1_Cチャンネル特徴マップ → L2_Cチャンネル特徴マップ（バッチ正規化あり, ドロップアウトあり（ドロップアウト率 0.2）, 活性化関数 Leaky-ReLU）
        # カーネルサイズ3x3，ストライド幅とゼロパディングは特徴マップの縦横幅が変わらないように自動設定
        self.conv2 = Conv(in_channels=L1_C, out_channels=L2_C, kernel_size=3, do_bn=True, dropout_ratio=0.2, activation='leaky_relu')
        # プーリング層2: maxプーリング
        self.pool2 = Pool(method='max')

        ### 3層目 ###

        # 畳込み層3: L2_Cチャンネル特徴マップ → L3_Cチャンネル特徴マップ（バッチ正規化あり, ドロップアウトなし, 活性化関数 ReLU）
        # カーネルサイズ3x3, ストライド幅とゼロパディングは敢えて手動設定
        self.conv3 = Conv(in_channels=L2_C, out_channels=L3_C, kernel_size=3, stride=1, padding=1, do_bn=True, activation='relu')
        # 平坦化層3
        self.flat3 = Flatten()

        ### 4層目 & 5層目 ###

        # 全結合層4: L3_C*(H/4)*(W/4)個のパーセプトロン → 2N個のパーセプトロン（バッチ正規化なし, ドロップアウトなし, 活性化関数ReLU）
        self.fc4 = FC(in_features=L3_C*(H//4)*(W//4), out_features=2*N, do_bn=False, activation='relu')
        # 全結合層5: 2N個のパーセプトロン → N個のパーセプトロン（バッチ正規化なし, ドロップアウトなし, 活性化関数なし）
        self.fc5 = FC(in_features=2*N, out_features=N, do_bn=False, activation='none')

    def forward(self, x):
        h = self.conv1(x) # 畳込み層1にデータを入力（この時点では特徴マップの縦幅・横幅は元のまま）
        h = self.pool1(h) # 続いてプーリング層1に通す（この時点で特徴マップの縦幅・横幅はそれぞれ元の 1/2 になる）
        h = self.conv2(h) # 続いて畳込み層2に通す（この時点では特徴マップの縦幅・横幅は元の 1/2 のまま）
        h = self.pool2(h) # 続いてプーリング層2に通す（この時点で特徴マップの縦幅・横幅はそれぞれ元の 1/4 になる）
        h = self.conv3(h) # 続いて畳込み層3に通す（この時点では特徴マップの縦幅・横幅は元の 1/4 のまま）
        h = self.flat3(h) # 続いて平坦化層3に通す（この時点で3次元的な構造を持つ特徴マップが一列に並べ直される）
        h = self.fc4(h)   # 続いて全結合層4に通す
        y = self.fc5(h)   # 最後に全結合層5に通す
        return y
