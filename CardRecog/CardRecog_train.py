import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
sys.path.append(os.path.abspath('..'))
import pickle
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from networks import CardClassifier
from mylib.visualizers import LossVisualizer
from mylib.data_io import CSVBasedDataset
from mylib.utility import print_args


# データセットファイル
DATASET_CSV_TRAIN = './card_images/train_list.csv'
DATASET_CSV_VALID = './card_images/test_list.csv'

# 画像ファイル名の先頭に付加する文字列（画像ファイルが存在するディレクトリのパス）
DATA_DIR = './card_images'

# 学習結果の保存先フォルダ
MODEL_DIR = './CNN_models'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pth')

# 画像のサイズ・チャンネル数
C = 3 # チャンネル数
H = 144 # 縦幅
W = 96  # 横幅


# デバイス, エポック数, バッチサイズなどをコマンドライン引数から取得し変数に保存
parser = argparse.ArgumentParser(description='Convolutional Neural Network Sample Code (training)')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU/CUDA ID (negative value indicates CPU)')
parser.add_argument('--epochs', '-e', default=20, type=int, help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', default=256, type=int, help='minibatch size')
parser.add_argument('--autosave', '-s', help='this option makes the model automatically saved in each epoch', action='store_true')
args = print_args(parser.parse_args())
DEVICE = args['device']
N_EPOCHS = args['epochs']
BATCH_SIZE = args['batchsize']
AUTO_SAVE = args['autosave']

# CSVファイルを読み込み, 訓練データセットを用意
train_dataset = CSVBasedDataset(
    filename = DATASET_CSV_TRAIN,
    items = [
        'File Path', # X
        'Class', # Y
    ],
    dtypes = [
        'image', # Xの型
        'label', # Yの型
    ],
    dirname = DATA_DIR,
)
train_size = len(train_dataset)
with open(os.path.join(MODEL_DIR, 'fdicts.pkl'), 'wb') as fdicts_file:
    pickle.dump(train_dataset.forward_dicts, fdicts_file)

# 同じく，検証用データセットを用意
valid_dataset = CSVBasedDataset(
    filename = DATASET_CSV_VALID,
    items = [
        'File Path', # X
        'Class', # Y
    ],
    dtypes = [
        'image', # Xの型
        'label', # Yの型
    ],
    dirname = DATA_DIR,
    fdicts = train_dataset.forward_dicts,
)
valid_size = len(valid_dataset)

# 訓練データおよび検証用データをミニバッチに分けて使用するための「データローダ」を用意
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ニューラルネットワークの作成
model = CardClassifier(C=C, H=H, W=W, N_CLASS=52).to(DEVICE) # カードの種類を判定するモデル

# 最適化アルゴリズムの指定（ここでは Adam を使用）
optimizer = optim.Adam(model.parameters())

# 損失関数：クロスエントロピー損失を使用
loss_func = nn.CrossEntropyLoss()
loss_viz = LossVisualizer(items=['train loss', 'valid loss'])

# 勾配降下法による繰り返し学習
for epoch in range(N_EPOCHS):

    print('Epoch {0}:'.format(epoch + 1))

    # 学習
    model.train()
    sum_loss = 0
    for X, Y in tqdm(train_dataloader):
        for param in model.parameters():
            param.grad = None
        X = X.to(DEVICE) # 画像（ミニバッチ）
        Y = Y.to(DEVICE) # クラスラベル（ミニバッチ）
        Y_pred = model(X) # ミニバッチ内の各カードの種類を判定
        loss = loss_func(Y_pred, Y) # 損失関数の現在値を計算
        loss.backward() # 誤差逆伝播法により，個々のパラメータに関する損失関数の勾配（偏微分）を計算
        optimizer.step() # 勾配に沿ってパラメータの値を更新
        sum_loss += float(loss) * len(X)
    avg_loss = sum_loss / train_size
    loss_viz.add_value('train loss', avg_loss)
    print('train loss = {0:.6f}'.format(avg_loss))

    # 検証
    model.eval()
    sum_loss = 0
    n_failed = 0
    with torch.inference_mode():
        for X, Y in tqdm(valid_dataloader):
            X = X.to(DEVICE) # 画像（ミニバッチ）
            Y = Y.to(DEVICE) # クラスラベル（ミニバッチ）
            Y_pred = model(X) # ミニバッチ内の各カードの種類を判定
            loss = loss_func(Y_pred, Y) # 損失関数の現在値を計算
            sum_loss += float(loss) * len(X)
            n_failed += torch.count_nonzero( # 推定値と正解値が一致していないデータの個数を数える
                torch.argmax(Y_pred, dim=1) - Y
            )
    avg_loss = sum_loss / valid_size
    accuracy = (valid_size - n_failed) / valid_size
    loss_viz.add_value('valid loss', avg_loss)
    loss_viz.show()
    print('valid loss = {0:.6f}'.format(avg_loss))
    print('accuracy = {0:.2f}%'.format(100 * accuracy))
    print('')

    # 学習途中のモデルの保存
    if AUTO_SAVE:
        torch.save(model.to('cpu').state_dict(), os.path.join(MODEL_DIR, 'autosaved_model_ep{0}.pth'.format(epoch + 1)))
        model.to(DEVICE)

# 学習結果のニューラルネットワークモデルをファイルに保存
torch.save(model.to('cpu').state_dict(), MODEL_PATH)
