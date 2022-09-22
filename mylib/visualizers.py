import torch
import numpy as np
import matplotlib.pyplot as plt


def lighten(c):
    return np.uint8(255 - (255 - c) // 2)


# 二次元データ識別器の可視化器
class ClassifierVisualizer():

    # コンストラクタ
    #   - n_classes: 可視化対象の識別器が扱うクラスの数
    #   - size: 可視化画像のサイズ
    #   - hrange: 横軸の最小値・最大値
    #   - vrange: 縦軸の最小値・最大値
    #   - title: 可視化グラフのタイトル
    #   - hlabel: 横軸の軸ラベル名
    #   - vlabel: 縦軸の軸ラベル名
    #   - clabels: クラスラベル名（凡例の表示に使用）
    #   - bins: 縦軸・横軸の目盛りを何段階用意するか
    def __init__(self, n_classes, size=512, hrange=(-1.0, 1.0), vrange=(-1.0, 1.0), title='title', hlabel='feature 1', vlabel='feature 2', clabels=None, bins=5):
        self.n_classes = n_classes
        self.size = size
        self.bins = bins
        self.hrange = hrange
        self.vrange = vrange
        self.title = title
        self.hlabel = hlabel
        self.vlabel = vlabel
        if clabels is None:
            self.clabels = []
            for i in range(n_classes):
                self.clabels.append('class {0}'.format(i + 1))
        else:
            self.clabels = clabels
        for i in range(0, size):
            y = np.ones((size, 1), dtype=np.float32) * i
            x = np.asarray([np.arange(size)], dtype=np.float32).transpose(1, 0)
            y = vrange[0] + y * (vrange[1] - vrange[0]) / (size - 1)
            x = hrange[0] + x * (hrange[1] - hrange[0]) / (size - 1)
            c = np.concatenate([x, y], axis=1)
            self.data = c if i == 0 else np.concatenate([self.data, c], axis=0)
        fig = plt.figure()
        self.ax = fig.add_subplot(1, 1, 1)

    # 可視化の実行
    #   - model: 識別器クラスのインスタンス
    #   - class_colors: 各クラスの表示色（RGBの順）
    #   - sec: 表示時間（秒数. 1 未満のときは 1 に丸め込む. デフォルトでは 1 秒）
    #   - samples: 可視化画像に重畳する実データ
    #              samples[0] がラベルデータ(一次元の numpy.ndarray, int32), 
    #              samples[1] が特徴量データ(二次元の numpy.ndarray, float32) となるように指定
    #   - その他の引数: コンストラクタを参照．変更したい場合のみ指定する
    def show(self, model, class_colors, sec=1, samples=None, title=None, hlabel=None, vlabel=None, clabels=None, bins=None):

        for param in model.parameters():
            device = param.data.device
            break

        self.ax.cla()

        # パラメータ値に変更がある場合は更新
        if title is not None: self.title = title
        if hlabel is not None: self.hlabel = hlabel
        if vlabel is not None: self.vlabel = vlabel
        if clabels is not None: self.clabels = clabels
        if bins is not None: self.bins = bins

        # グラフタイトルの設定
        self.ax.set_title(self.title)

        # 背景画像の作成
        model.eval()
        result = torch.argmax(model(torch.tensor(self.data, device=device)), dim=1)
        result = result.to('cpu').detach().numpy().copy().reshape((self.size, self.size, 1))
        r = np.zeros((self.size, self.size, 1), dtype=np.uint8)
        g = np.zeros((self.size, self.size, 1), dtype=np.uint8)
        b = np.zeros((self.size, self.size, 1), dtype=np.uint8)
        for i in range(self.n_classes):
            r += np.where(result == i, lighten(class_colors[i][0]), np.uint8(0))
            g += np.where(result == i, lighten(class_colors[i][1]), np.uint8(0))
            b += np.where(result == i, lighten(class_colors[i][2]), np.uint8(0))
        img = np.concatenate([r, g, b], axis=2)
        del result

        # 背景画像の設定
        self.ax.imshow(img)

        # 実データの描画
        if samples is not None:
            lab = np.asarray(samples[1])
            feat = np.asarray(samples[0])
            ptx = np.floor((self.size - 1) * (feat[:,0] - self.hrange[0]) / (self.hrange[1] - self.hrange[0])).astype(np.int32)
            pty = np.floor((self.size - 1) * (feat[:,1] - self.vrange[0]) / (self.vrange[1] - self.vrange[0])).astype(np.int32)
            for i in range(self.n_classes):
                ccolor = np.asarray(class_colors[i]).reshape((1, 3)) / 255
                self.ax.scatter(ptx[lab==i], pty[lab==i], c=ccolor, label=self.clabels[i])
            self.ax.legend(loc='best')

        # 縦軸・横軸の目盛りの作成
        self.ax.set_xticks(np.linspace(0, self.size-1, self.bins))
        self.ax.set_yticks(np.linspace(0, self.size-1, self.bins))
        hlabels = []
        vlabels = []
        for i in range(0, self.bins):
            hlabels.append(format(self.hrange[0] + i * (self.hrange[1] - self.hrange[0]) / (self.bins - 1), '.3g'))
            vlabels.append(format(self.vrange[0] + i * (self.vrange[1] - self.vrange[0]) / (self.bins - 1), '.3g'))
        self.ax.set_xticklabels(hlabels)
        self.ax.set_yticklabels(vlabels)
        self.ax.grid(True)

        # 軸ラベルを設定
        self.ax.set_xlabel(self.hlabel)
        self.ax.set_ylabel(self.vlabel)

        # 縦軸の上下を反転
        self.ax.invert_yaxis()

        # 表示（1秒後にウィンドウを閉じる）
        plt.pause(sec)
