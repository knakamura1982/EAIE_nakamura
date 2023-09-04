import os
import sys
sys.path.append(os.path.abspath('..'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from mylib.basic_layers import *


# 「プレイヤースコア」「手札の枚数」の2情報から4種類の行動の選択確率を計算するニューラルネットワーク
class BJNet(nn.Module):

    def __init__(self):
        super(BJNet, self).__init__()

        # 1層目: 2次元入力 → 10パーセプトロン（バッチ正規化あり, ドロップアウトなし, 活性化関数 ReLU）
        self.layer1 = FC(in_features=2, out_features=10, do_bn=True, activation='relu')

        # 2層目: 10パーセプトロン → 10パーセプトロン（バッチ正規化なし, ドロップアウトあり（ドロップアウト率 0.5）, 活性化関数 Tanh）
        self.layer2 = FC(in_features=10, out_features=10, do_bn=False, dropout_ratio=0.5, activation='tanh')

        # 3層目: 10パーセプトロン → 4クラス出力（バッチ正規化なし, ドロップアウトなし, 活性化関数なし）
        self.layer3 = FC(in_features=10, out_features=5, do_bn=False, activation='none')

    def forward(self, x):
        h = self.layer1(x) # 1層目にデータを入力
        h = self.layer2(h) # 続いて2層目に通す
        y = self.layer3(h) # 最後に3層目に通す
        return y
