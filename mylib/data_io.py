import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from matplotlib import cm
from torch.utils.data import Dataset


# 整数 i を n 次元の one-hot 表現に変換
def __to_one_hot__(i, n):
    a = np.zeros(n, dtype=np.int32)
    a[i] = 1
    return a

# カテゴリカルデータのリストを整数値（np.int32）のリストに変換する関数
def __to_numerical__(cat_data, one_hot=False, fdict=None):

    # まず, データ値の種類数を求めておく
    s = sorted(set(cat_data))
    n = len(s)

    # 次に，正引き／逆引き辞書を作る
    if fdict is None:
        if one_hot:
            forward_dict = {item:__to_one_hot__(i, n) for i, item in enumerate(s)}
        else:
            forward_dict = {item:i for i, item in enumerate(s)}
    else:
        forward_dict = fdict
    reverse_dict = [s[i] for i in range(n)]

    # 変換
    data = np.asarray(list(map(forward_dict.get, cat_data)), dtype=np.int32)

    return data, forward_dict, reverse_dict


# データセット読込用のクラス
#   - filename: 読み込む csv ファイルのファイルパス
#   - items: データとして読み込む列の項目名
#   - dtypes: データの型（ 'float', 'label', 'one-hot', 'image' のいずれか）
#   - fdicts: カテゴリデータを整数値に変換するための正引き辞書（Noneの場合は自動作成）
#   - dirname: データ型が 'image' のとき, ファイル名の先頭に付加するディレクトリ名を指定するのに使用
#   - img_mode: データ型が 'image' のとき, カラー画像か否かを指定するのに使用（ 'color' か 'grayscale' のいずれか）
class CSVBasedDataset(Dataset):

    # コンストラクタ
    def __init__(self, filename, items, dtypes, fdicts=None, dirname='./', img_mode=''):
        super(CSVBasedDataset, self).__init__()

        self.dtypes = dtypes
        self.dirname = dirname
        if img_mode == 'color':
            self.img_mode = torchvision.io.image.ImageReadMode.RGB
        elif img_mode == 'grayscale':
            self.img_mode = torchvision.io.image.ImageReadMode.GRAY
        else:
            self.img_mode = torchvision.io.image.ImageReadMode.UNCHANGED

        # csv ファイルを読み込む
        df = pd.read_csv(filename)

        # items で指定された列のみをメンバ変数に保存
        self.data = []
        self.forward_dicts = []
        self.reverse_dicts = []
        for i in range(len(items)):
            fd = None
            rd = None
            if dtypes[i] == 'float':
                X = torch.tensor(df[items[i]].values, dtype=torch.float32, device='cpu')
            elif dtypes[i] == 'label':
                if type(items[i]) is list:
                    X = []
                    fd = []
                    rd = []
                    j = 0
                    for item in items[i]:
                        if fdicts is None:
                            X_temp, fd_temp, rd_temp = __to_numerical__(df[item].to_list(), one_hot=False)
                        else:
                            X_temp, fd_temp, rd_temp = __to_numerical__(df[item].to_list(), one_hot=False, fdict=fdicts[i][j])
                        X.append(X_temp)
                        fd.append(fd_temp)
                        rd.append(rd_temp)
                        j += 1
                    X = np.concatenate(X, axis=1)
                else:
                    if fdicts is None:
                        X, fd, rd = __to_numerical__(df[items[i]].to_list(), one_hot=False)
                    else:
                        X, fd, rd = __to_numerical__(df[items[i]].to_list(), one_hot=False, fdict=fdicts[i])
                X = torch.tensor(X, dtype=torch.long, device='cpu')
            elif dtypes[i] == 'one-hot':
                if type(items[i]) is list:
                    X = []
                    fd = []
                    rd = []
                    j = 0
                    for item in items[i]:
                        if fdicts is None:
                            X_temp, fd_temp, rd_temp = __to_numerical__(df[item].to_list(), one_hot=True)
                        else:
                            X_temp, fd_temp, rd_temp = __to_numerical__(df[item].to_list(), one_hot=True, fdict=fdicts[i][j])
                        X.append(X_temp)
                        fd.append(fd_temp)
                        rd.append(rd_temp)
                        j += 1
                    X = np.concatenate(X, axis=1)
                else:
                    if fdicts is None:
                        X, fd, rd = __to_numerical__(df[items[i]].to_list(), one_hot=True)
                    else:
                        X, fd, rd = __to_numerical__(df[items[i]].to_list(), one_hot=True, fdict=fdicts[i])
                X = torch.tensor(X, dtype=torch.float32, device='cpu')
            elif dtypes[i] == 'image':
                X = df[items[i]].to_list()
            else:
                continue
            self.data.append(X)
            self.forward_dicts.append(fd)
            self.reverse_dicts.append(rd)

        # データセットサイズ（データ数）を記憶しておく => int型のメンバ変数 len に保存
        self.len = len(self.data[0])

    # データセットサイズを返却する関数
    def __len__(self):
        return self.len

    # index 番目のデータを返却する関数
    # データローダは，この関数を必要な回数だけ呼び出して，自動的にミニバッチを作成してくれる
    def __getitem__(self, index):
        single_data = []
        for i in range(len(self.data)):
            if self.dtypes[i] == 'image':
                # 実際に画像ファイルを読み込み，画素値を 0～1 に正規化する（元々が 0~255 なので, 255 で割る）
                x = torchvision.io.read_image(os.path.join(self.dirname, self.data[i][index]), mode=self.img_mode) / 255
            else:
                x = self.data[i][index]
            single_data.append(x)
        if len(self.data) == 1:
            return single_data[0]
        else:
            return tuple(single_data)


# 画像表示用関数
#   - data: 表示対象データ（画素値は 0～1 に正規化されていることを想定）
#   - title: 表示用ウィンドウのタイトル
#   - sec: 何秒間表示するか（ 0 以下の場合は「閉じる」ボタンが押されるまで継続表示，デフォルトでは 0 ）
#   - save_fig: True なら表示結果をファイルにも保存する（デフォルトでは False ）
#   - save_only: True ならファイルに保存するだけで表示しない（デフォルトでは False ）
#   - save_dir: ファイルに保存する場合の保存先ディレクトリ（デフォルトではプログラムの実行ディレクトリになる）
def show_single_image(data, title='no_title', sec=0, save_fig=False, save_only=False, save_dir='./'):
    img = np.asarray(data) # 入力データ（配列）の型を numpy.ndarray に変更
    if len(img.shape) == 4:
        img = img[0].transpose(1, 2, 0) # 入力データが4次元配列のときは, 先頭の1枚のみを抽出する
    elif len(img.shape) == 3:
        if img.shape[0] == 1 or img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
    elif len(img.shape) == 2:
        img = img.reshape((*img.shape, 1)) # 入力データが2次元配列のときは, 3次元化（チャンネル数:1）
    img = (255 * np.minimum(np.ones(img.shape), np.maximum(np.zeros(img.shape), img))).astype(np.uint8) # 画素値を 0～1 に丸め込んだ後で 255 倍する
    plt.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)
    plt.axis('off')
    plt.title(title)
    plt.imshow(img, cmap=cm.gray, interpolation='nearest')
    if save_fig or save_only:
        plt.savefig(os.path.join(save_dir, title + '.png'), bbox_inches='tight')
    if not save_only:
        if sec <= 0:
            plt.show()
        else:
            plt.pause(sec)
    plt.close()


# 複数枚の画像を一度に表示する関数
#   - data: 表示対象データ（バッチサイズxチャンネル数x縦幅x横幅の 4 次元テンソル, 画素値は 0～1 に正規化されていることを想定）
#   - num: 表示する画像の枚数
#   - num_per_row: 1行あたりの表示枚数（デフォルトでは sqrt(num) くらい）
#   - title: 表示用ウィンドウのタイトル
#   - sec: 何秒間表示するか（ 0 以下の場合は「閉じる」ボタンが押されるまで継続表示，デフォルトでは 0 ）
#   - save_fig: True なら表示結果をファイルにも保存する（デフォルトでは False ）
#   - save_only: True ならファイルに保存するだけで表示しない（デフォルトでは False ）
#   - save_dir: ファイルに保存する場合の保存先ディレクトリ（デフォルトではプログラムの実行ディレクトリになる）
def show_images(data, num, num_per_row=0, title='no_title', sec=0, save_fig=False, save_only=False, save_dir='./'):
    if num_per_row <= 0:
        num_per_row = int(np.ceil(np.sqrt(num)))
    data = np.asarray(data)
    data = (255 * np.minimum(np.ones(data.shape), np.maximum(np.zeros(data.shape), data))).astype(np.uint8)
    n_total = min(data.shape[0], num) # 保存するデータの総数
    n_rows = int(np.ceil(n_total / num_per_row)) # 保存先画像においてデータを何行に分けて表示するか
    plt.figure(title, figsize=(1 + num_per_row * data.shape[3] / 128, 1 + n_rows * data.shape[2] / 128))
    plt.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)
    for i in range(0, n_total):
        plt.subplot(n_rows, num_per_row, i+1)
        plt.axis('off')
        plt.imshow(data[i].transpose(1, 2, 0), cmap=cm.gray, interpolation='nearest')
    if save_fig or save_only:
        plt.savefig(os.path.join(save_dir, title + '.png'), bbox_inches='tight')
    if not save_only:
        if sec <= 0:
            plt.show()
        else:
            plt.pause(sec)
    plt.close()


# 画素値の範囲を [0, 1] から [-1, 1] に変更
def to_tanh_image(img):
    return 2 * img - 1

# 画素値の範囲を [-1, 1] から [0, 1] に変更
def to_sigmoid_image(img):
    return 0.5 * (img + 1)
