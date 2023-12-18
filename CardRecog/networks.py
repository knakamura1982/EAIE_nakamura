import os
import sys
sys.path.append(os.path.abspath('..'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from mylib.basic_layers import *


# 単純にカードの種類を認識するCNN
class CardClassifier(nn.Module):

    # C: 入力画像のチャンネル数（グレースケール画像なら1，カラー画像なら3）
    # H: 入力画像の縦幅（8の倍数を想定）
    # W: 入力画像の横幅（8の倍数を想定）
    # N_CLASS: カードの種類数
    def __init__(self, C, H, W, N_CLASS=52):
        super(CardClassifier, self).__init__()

        # 層ごとのチャンネル数
        L1_C = 4  # 1層目
        L2_C = 8  # 2層目
        L3_C = 16 # 3層目
        L4_N = 32 # 4層目（4層目は畳み込み層ではなく全結合層であり，L4_Nはそのユニット数を表す）

        ### 1～3層目: スート認識と数字認識で共通

        # 1層目: 畳込み, Cチャンネル入力 → L1_Cチャンネル，カーネルサイズ5x5, ストライド幅とゼロパディングは特徴マップの縦横幅が変わらないように設定
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=L1_C, kernel_size=5, stride=1, padding=2, bias=False), # 畳込み
            nn.BatchNorm2d(num_features=L1_C), # バッチ正規化（上の畳込み層の出力チャンネル数がL1_Cなので）
            nn.LeakyReLU(), # 活性化関数 Leaky-ReLU
            nn.AvgPool2d(kernel_size=2, stride=2), # 平均値プーリング
        )

        # 2層目: 畳込み, L1_Cチャンネル → L2_Cチャンネル，カーネルサイズ3x3, ストライド幅とゼロパディングは特徴マップの縦横幅が変わらないように設定
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=L1_C, out_channels=L2_C, kernel_size=3, stride=1, padding=1, bias=False), # 畳込み
            nn.BatchNorm2d(num_features=L2_C), # バッチ正規化（前の畳込み層の出力チャンネル数がL2_Cなので）
            nn.ReLU(), # 活性化関数 ReLU
            nn.MaxPool2d(kernel_size=2, stride=2), # maxプーリング
        )

        # 3層目: 畳込み, L2_Cチャンネル → L3_Cチャンネル，カーネルサイズ3x3, ストライド幅とゼロパディングは特徴マップの縦横幅が変わらないように設定
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=L2_C, out_channels=L3_C, kernel_size=3, stride=1, padding=1, bias=False), # 畳込み
            nn.BatchNorm2d(num_features=L3_C), # バッチ正規化（前の畳込み層の出力チャンネル数がL2_Cなので）
            nn.ReLU(), # 活性化関数 ReLU
            nn.MaxPool2d(kernel_size=2, stride=2), # maxプーリング
        )

        # 平坦化層
        self.flatten = nn.Flatten()

        # 4層目: 全結合, L3_C*(H/8)*(W/8)個のパーセプトロン → L4_N個のパーセプトロン
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=L3_C*(H//8)*(W//8), out_features=L4_N, bias=False), # 全結合
            nn.BatchNorm1d(num_features=L4_N), # バッチ正規化（前の畳込み層の出力ユニット数がL4_Nなので）
            nn.ReLU(), # 活性化関数 ReLU
        )

        # 5層目: 全結合, L4_N個のパーセプトロン → N_SUITS個のパーセプトロン
        self.layer5 = nn.Linear(in_features=L4_N, out_features=N_CLASS)

    def forward(self, x):

        ### 共通部分 ###
        h = self.layer1(x)  # 1層目
        h = self.layer2(h)  # 2層目
        h = self.layer3(h)  # 3層目
        h = self.flatten(h) # 平坦化層
        h = self.layer4(h)  # 4層目
        y = self.layer5(h)  # 5層目

        ### 認識結果を返却 ###
        return y


# カードの数字とスートを個別に認識するCNN
class CardClassifierAlt(nn.Module):

    # C: 入力画像のチャンネル数（グレースケール画像なら1，カラー画像なら3）
    # H: 入力画像の縦幅（8の倍数を想定）
    # W: 入力画像の横幅（8の倍数を想定）
    # N_SUITS: 認識対象のスートの種類数
    # N_NUMBERS: 認識対象の数字の種類数
    def __init__(self, C, H, W, N_SUITS=4, N_NUMBERS=13):
        super(CardClassifierAlt, self).__init__()

        # 層ごとのチャンネル数
        L1_C = 4  # 1層目
        L2_C = 8  # 2層目
        L3_C = 16 # 3層目
        L4_N = 16 # 4層目（4層目は畳み込み層ではなく全結合層であり，L4_Nはそのユニット数を表す）

        ### 1～3層目: スート認識と数字認識で共通

        # 1層目: 畳込み, Cチャンネル入力 → L1_Cチャンネル，カーネルサイズ5x5, ストライド幅とゼロパディングは特徴マップの縦横幅が変わらないように設定
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=L1_C, kernel_size=5, stride=1, padding=2, bias=False), # 畳込み
            nn.BatchNorm2d(num_features=L1_C), # バッチ正規化（上の畳込み層の出力チャンネル数がL1_Cなので）
            nn.LeakyReLU(), # 活性化関数 Leaky-ReLU
            nn.AvgPool2d(kernel_size=2, stride=2), # 平均値プーリング
        )

        # 2層目: 畳込み, L1_Cチャンネル → L2_Cチャンネル，カーネルサイズ3x3, ストライド幅とゼロパディングは特徴マップの縦横幅が変わらないように設定
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=L1_C, out_channels=L2_C, kernel_size=3, stride=1, padding=1, bias=False), # 畳込み
            nn.BatchNorm2d(num_features=L2_C), # バッチ正規化（前の畳込み層の出力チャンネル数がL2_Cなので）
            nn.ReLU(), # 活性化関数 ReLU
            nn.MaxPool2d(kernel_size=2, stride=2), # maxプーリング
        )

        # 3層目: 畳込み, L2_Cチャンネル → L3_Cチャンネル，カーネルサイズ3x3, ストライド幅とゼロパディングは特徴マップの縦横幅が変わらないように設定
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=L2_C, out_channels=L3_C, kernel_size=3, stride=1, padding=1, bias=False), # 畳込み
            nn.BatchNorm2d(num_features=L3_C), # バッチ正規化（前の畳込み層の出力チャンネル数がL2_Cなので）
            nn.ReLU(), # 活性化関数 ReLU
            nn.MaxPool2d(kernel_size=2, stride=2), # maxプーリング
        )

        # 平坦化層
        self.flatten = nn.Flatten()

        ### 4, 5層目: スート認識用 ###

        # 4層目: 全結合, L3_C*(H/8)*(W/8)個のパーセプトロン → L4_N個のパーセプトロン
        self.layer4_s = nn.Sequential(
            nn.Linear(in_features=L3_C*(H//8)*(W//8), out_features=L4_N, bias=False), # 全結合
            nn.BatchNorm1d(num_features=L4_N), # バッチ正規化（前の畳込み層の出力ユニット数がL4_Nなので）
            nn.ReLU(), # 活性化関数 ReLU
        )

        # 5層目: 全結合, L4_N個のパーセプトロン → N_SUITS個のパーセプトロン
        self.layer5_s = nn.Linear(in_features=L4_N, out_features=N_SUITS)

        ### 4, 5層目: 数字認識用 ###

        # 4層目: 全結合, L3_C*(H/8)*(W/8)個のパーセプトロン → L4_N個のパーセプトロン
        self.layer4_n = nn.Sequential(
            nn.Linear(in_features=L3_C*(H//8)*(W//8), out_features=L4_N, bias=False), # 全結合
            nn.BatchNorm1d(num_features=L4_N), # バッチ正規化（前の畳込み層の出力ユニット数がL4_Nなので）
            nn.ReLU(), # 活性化関数 ReLU
        )

        # 5層目: 全結合, L4_N個のパーセプトロン → N_NUMBERS個のパーセプトロン
        self.layer5_n = nn.Linear(in_features=L4_N, out_features=N_NUMBERS)

    def forward(self, x):

        ### 共通部分 ###
        h = self.layer1(x)  # 1層目
        h = self.layer2(h)  # 2層目
        h = self.layer3(h)  # 3層目
        h = self.flatten(h) # 平坦化層

        ### ここから分岐．まずはスート認識部分 ###
        hs = self.layer4_s(h)  # 4層目
        ys = self.layer5_s(hs) # 5層目

        ### 続いて数字認識部分 ###
        hn = self.layer4_n(h)  # 4層目
        yn = self.layer5_n(hn) # 5層目

        ### 認識結果を返却 ###
        return ys, yn


# カードが絵札か否かの二クラス分類を実行するCNN
class CardChecker(nn.Module):

    # C: 入力画像のチャンネル数（グレースケール画像なら1，カラー画像なら3）
    # H: 入力画像の縦幅（8の倍数を想定）
    # W: 入力画像の横幅（8の倍数を想定）
    def __init__(self, C, H, W):
        super(CardChecker, self).__init__()

        # 層ごとのチャンネル数
        L1_C = 4 # 1層目
        L2_C = 4 # 2層目
        L3_C = 8 # 3層目
        L4_N = 8 # 4層目（4層目は畳み込み層ではなく全結合層であり，L4_Nはそのユニット数を表す）

        # 1層目: 畳込み, Cチャンネル入力 → L1_Cチャンネル，カーネルサイズ5x5, ストライド幅とゼロパディングは特徴マップの縦横幅が変わらないように設定
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=L1_C, kernel_size=5, stride=1, padding=2, bias=False), # 畳込み
            nn.BatchNorm2d(num_features=L1_C), # バッチ正規化（上の畳込み層の出力チャンネル数がL1_Cなので）
            nn.LeakyReLU(), # 活性化関数 Leaky-ReLU
            nn.AvgPool2d(kernel_size=2, stride=2), # 平均値プーリング
        )

        # 2層目: 畳込み, L1_Cチャンネル → L2_Cチャンネル，カーネルサイズ3x3, ストライド幅とゼロパディングは特徴マップの縦横幅が変わらないように設定
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=L1_C, out_channels=L2_C, kernel_size=3, stride=1, padding=1, bias=False), # 畳込み
            nn.BatchNorm2d(num_features=L2_C), # バッチ正規化（前の畳込み層の出力チャンネル数がL2_Cなので）
            nn.ReLU(), # 活性化関数 ReLU
            nn.MaxPool2d(kernel_size=2, stride=2), # maxプーリング
        )

        # 3層目: 畳込み, L2_Cチャンネル → L3_Cチャンネル，カーネルサイズ3x3, ストライド幅とゼロパディングは特徴マップの縦横幅が変わらないように設定
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=L2_C, out_channels=L3_C, kernel_size=3, stride=1, padding=1, bias=False), # 畳込み
            nn.BatchNorm2d(num_features=L3_C), # バッチ正規化（前の畳込み層の出力チャンネル数がL2_Cなので）
            nn.ReLU(), # 活性化関数 ReLU
            nn.MaxPool2d(kernel_size=2, stride=2), # maxプーリング
        )

        # 平坦化層
        self.flatten = nn.Flatten()

        # 4層目: 全結合, L3_C*(H/8)*(W/8)個のパーセプトロン → L4_N個のパーセプトロン
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=L3_C*(H//8)*(W//8), out_features=L4_N, bias=False), # 全結合
            nn.BatchNorm1d(num_features=L4_N), # バッチ正規化（前の畳込み層の出力ユニット数がL4_Nなので）
            nn.ReLU(), # 活性化関数 ReLU
        )

        # 5層目: 全結合, L4_N個のパーセプトロン → 2個のパーセプトロン
        self.layer5 = nn.Linear(in_features=L4_N, out_features=2)

    def forward(self, x):
        h = self.layer1(x)  # 1層目
        h = self.layer2(h)  # 2層目
        h = self.layer3(h)  # 3層目
        h = self.flatten(h) # 平坦化層
        h = self.layer4(h)  # 4層目
        y = self.layer5(h)  # 5層目
        return y
