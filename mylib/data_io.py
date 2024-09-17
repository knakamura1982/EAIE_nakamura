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


# 配列 arr から target_indices で指定された番号のみを抜き出す
def __extract__(arr, target_indices):
    return_value = [item for i, item in enumerate(arr) if i in target_indices]
    if type(arr) == np.ndarray:
        return np.asarray(return_value, dtype=arr.dtype)
    elif type(arr) == torch.tensor:
        return torch.tensor(return_value, dtype=arr.dtype, device=arr.device)
    else:
        return return_value


# データセット読込用のクラス
#   - filename: 読み込む csv ファイルのファイルパス
#   - items: データとして読み込む列の項目名
#   - dtypes: データの型（ 'float', 'label', 'one-hot', 'image' のいずれか）
#   - target_indices: 読み込み対象とするインデックスの集合（Noneの場合は全データを読み込む. デフォルトではNone）
#   - fdicts: カテゴリデータを整数値に変換するための正引き辞書（Noneの場合は自動作成）
#   - dirname: データ型が 'image' のとき, ファイル名の先頭に付加するディレクトリ名を指定するのに使用
#   - img_mode: データ型が 'image' のとき, カラー画像か否かを指定するのに使用（ 'color' か 'grayscale' のいずれか）
#   - img_range: データ型が 'image' のとき, 画素値をどの範囲の値に正規化するか（ [0, 1] または [-1, 1] を想定しているが，それ以外の範囲も指定可能）
#   - img_transform: データ型が 'image' のとき, 前処理として使用する Transform オブジェクト
class CSVBasedDataset(Dataset):

    # コンストラクタ
    def __init__(self, filename, items, dtypes, target_indices=None, fdicts=None, dirname='./', img_mode='', img_range=[0, 1], img_transform=None):
        super(CSVBasedDataset, self).__init__()

        self.dtypes = dtypes
        self.dirname = dirname
        self.img_range = img_range
        self.img_range_coeff = (self.img_range[1] - self.img_range[0]) / 255
        self.img_transform = img_transform
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
            if target_indices is not None:
                X = __extract__(X, target_indices)
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
                # 実際に画像ファイルを読み込み，画素値を正規化する
                x = torchvision.io.read_image(os.path.join(self.dirname, self.data[i][index]), mode=self.img_mode)
                x = x * self.img_range_coeff + self.img_range[0]
                if self.img_transform is not None:
                    x = self.img_transform(x)
            else:
                x = self.data[i][index]
            single_data.append(x)
        if len(self.data) == 1:
            return single_data[0]
        else:
            return tuple(single_data)


# 画像データを triplet（三つ組）として読み込むためのデータセット読み出しクラス
#   - filename: 読み込む csv ファイルのファイルパス
#   - data_item: データとして読み込む列（画像ファイルパス）の項目名
#   - label_item: ラベルとして読み込む列（物体クラス名，人物IDなど）の項目名
#   - target_indices: 読み込み対象とするインデックスの集合（Noneの場合は全データを読み込む. デフォルトではNone）
#   - target_labels: 読み込み対象とするラベルの集合（学習データと検証データをラベル単位で切り分ける際に使用．デフォルトでは無視される）
#   - use_anchor_label: Anchorデータのラベル情報を同時に使用するか否か
#   - fdict: ラベルを整数値に変換するための正引き辞書（Noneの場合は自動作成）
#   - dirname: データ型が 'image' のとき, ファイル名の先頭に付加するディレクトリ名を指定するのに使用
#   - img_mode: データ型が 'image' のとき, カラー画像か否かを指定するのに使用（ 'color' か 'grayscale' のいずれか）
#   - img_range: データ型が 'image' のとき, 画素値をどの範囲の値に正規化するか（ [0, 1] または [-1, 1] を想定しているが，それ以外の範囲も指定可能）
#   - img_transform: データ型が 'image' のとき, 前処理として使用する Transform オブジェクト
class TripletImageDataset(Dataset):

    # コンストラクタ
    def __init__(self, filename, data_item, label_item, target_indices=None, target_labels=None, use_anchor_label=False, fdict=None, dirname='./', img_mode='', img_range=[0, 1], img_transform=None):
        super(TripletImageDataset, self).__init__()

        self.use_anchor_label = use_anchor_label
        self.dirname = dirname
        self.img_range = img_range
        self.img_range_coeff = (self.img_range[1] - self.img_range[0]) / 255
        self.img_transform = img_transform
        if img_mode == 'color':
            self.img_mode = torchvision.io.image.ImageReadMode.RGB
        elif img_mode == 'grayscale':
            self.img_mode = torchvision.io.image.ImageReadMode.GRAY
        else:
            self.img_mode = torchvision.io.image.ImageReadMode.UNCHANGED

        # csv ファイルを読み込む
        df = pd.read_csv(filename)

        # 画像ファイル名リストおよびラベルリストを取得
        if target_labels is None:
            self.data = df[data_item].to_list()
            self.label = df[label_item].to_list()
            if fdict is None:
                self.label, self.fdict, self.rdict = __to_numerical__(self.label, one_hot=False)
            else:
                self.label, self.fdict, self.rdict = __to_numerical__(self.label, one_hot=False, fdict=fdict)
        else:
            temp_data = df[data_item].to_list()
            temp_label = df[label_item].to_list()
            self.data = [ temp_data[i] for i in range(len(temp_label)) if temp_label[i] in target_labels ]
            self.label = [ temp_label[i] for i in range(len(temp_label)) if temp_label[i] in target_labels ]
            if fdict is None:
                self.label, self.fdict, self.rdict = __to_numerical__(self.label, one_hot=False)
            else:
                self.label, self.fdict, self.rdict = __to_numerical__(self.label, one_hot=False, fdict=fdict)
            del temp_data
            del temp_label
        if target_indices is not None:
            self.data = __extract__(self.data, target_indices)
            self.label = __extract__(self.label, target_indices)

        # データセットサイズ（データ数）を記憶しておく => int型のメンバ変数 len に保存
        self.len = len(self.data)

    # データセットサイズを返却する関数
    def __len__(self):
        return self.len

    # index 番目のデータを返却する関数
    # データローダは，この関数を必要な回数だけ呼び出して，自動的にミニバッチを作成してくれる
    def __getitem__(self, index):

        # index 番目の画像を anchor として，positive と negative をランダム選択
        lab = self.label[index]
        p_index = index # 同じラベルの画像が 1 枚しかない場合は，仕方がないので anchor == positive を許容する
        p_cands = np.where(self.label == lab)[0]
        if len(p_cands) >= 2:
            while p_index == index:
                p_index = np.random.choice(p_cands)
        n_index = np.random.choice(np.where(self.label != lab)[0])

        # 実際に画像ファイルを読み込み，画素値を正規化する
        anc = torchvision.io.read_image(os.path.join(self.dirname, self.data[index]), mode=self.img_mode)
        pos = torchvision.io.read_image(os.path.join(self.dirname, self.data[p_index]), mode=self.img_mode)
        neg = torchvision.io.read_image(os.path.join(self.dirname, self.data[n_index]), mode=self.img_mode)
        anc = anc * self.img_range_coeff + self.img_range[0]
        pos = pos * self.img_range_coeff + self.img_range[0]
        neg = neg * self.img_range_coeff + self.img_range[0]
        if self.img_transform is not None:
            anc = self.img_transform(anc)
            pos = self.img_transform(pos)
            neg = self.img_transform(neg)

        if self.use_anchor_label:
            return anc, pos, neg, torch.tensor(lab, dtype=torch.long)
        else:
            return anc, pos, neg


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


# モデル保存用にファイル名を調整する
#   - base: 基本となるファイル名
#   - ep: エポック番号
def autosaved_model_name(base: str, ep: int):
    p = base.rfind('.')
    return '{0}_ep{1}.pth'.format(base[:p], ep)
