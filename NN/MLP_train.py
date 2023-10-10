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
from torch.utils.data import DataLoader, random_split
from networks import SampleMLP, myMLP
from mylib.data_io import CSVBasedDataset
from mylib.visualizers import ClassifierVisualizer
from mylib.utility import print_args


# データセットファイル
DATASET_CSV = './csv_data/weather_train.csv'

# 学習結果の保存先フォルダ
MODEL_DIR = './MLP_models'

# 学習過程を可視化するか否か
# 入力が2次元ベクトルでないときにTrueを指定するとエラーになる
VISUALIZE = True


# デバイス, エポック数, バッチサイズなどをコマンドライン引数から取得し変数に保存
parser = argparse.ArgumentParser(description='Multi-Layer Perceptron Sample Code (training)')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU/CUDA ID (negative value indicates CPU)')
parser.add_argument('--epochs', '-e', default=50, type=int, help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', default=100, type=int, help='minibatch size')
parser.add_argument('--model', '-m', default=os.path.join(MODEL_DIR, 'model.pth'), type=str, help='file path of trained model')
parser.add_argument('--autosave', '-s', help='this option makes the model automatically saved in each epoch', action='store_true')
args = print_args(parser.parse_args())
DEVICE = args['device']
N_EPOCHS = args['epochs']
BATCH_SIZE = args['batchsize']
MODEL_PATH = args['model']
AUTO_SAVE = args['autosave']

# CSVファイルを読み込み, 訓練データセットを用意
dataset = CSVBasedDataset(
    filename = DATASET_CSV,
    items = [
        ['平均気温', '平均湿度'], # X
        '天気概況' # Y
    ],
    dtypes = [
        'float', # Xの型
        'label' # Yの型
    ],
)
with open(os.path.join(MODEL_DIR, 'fdicts.pkl'), 'wb') as fdicts_file:
    pickle.dump(dataset.forward_dicts, fdicts_file)

# 訓練データセットを分割し，一方を検証用に回す
dataset_size = len(dataset)
valid_size = int(0.05 * dataset_size) # 全体の 5% を検証用に
train_size = dataset_size - valid_size # 残りの 95% を学習用に
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

# 訓練データおよび検証用データをミニバッチに分けて使用するための「データローダ」を用意
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# 学習経過を可視化する準備
if VISUALIZE:
    visualizer = ClassifierVisualizer(
        n_classes = 3,
        clabels = ['sunny', 'cloudy', 'rainy'],
        hrange = [-5, 40],
        vrange = [20, 110],
        hlabel = 'temperature (degree)',
        vlabel = 'humidity (%)',
        bins = 10
    )
    samples_for_visualization = train_dataset[:200]

# ニューラルネットワークの作成
model = SampleMLP().to(DEVICE)
#model = myMLP().to(DEVICE) # myMLPクラスを用いる場合はこちらを使用

# 最適化アルゴリズムの指定（ここでは Adam を使用）
optimizer = optim.Adam(model.parameters())

# 損失関数：クロスエントロピー損失を使用
loss_func = nn.CrossEntropyLoss()

# 勾配降下法による繰り返し学習
for epoch in range(N_EPOCHS):

    print('Epoch {0}:'.format(epoch + 1))

    # 学習
    model.train()
    sum_loss = 0
    for X, Y in tqdm(train_dataloader):
        for param in model.parameters():
            param.grad = None
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)
        Y_pred = model(X) # 入力値 X を現在のニューラルネットワークに入力し，出力の推定値を得る
        loss = loss_func(Y_pred, Y) # 損失関数の現在値を計算
        loss.backward() # 誤差逆伝播法により，個々のパラメータに関する損失関数の勾配（偏微分）を計算
        optimizer.step() # 勾配に沿ってパラメータの値を更新
        sum_loss += float(loss) * len(X)
    avg_loss = sum_loss / train_size
    print('train loss = {0:.6f}'.format(avg_loss))

    # 検証
    model.eval()
    sum_loss = 0
    n_failed = 0
    with torch.inference_mode():
        for X, Y in tqdm(valid_dataloader):
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)
            Y_pred = model(X)
            loss = loss_func(Y_pred, Y)
            sum_loss += float(loss) * len(X)
            n_failed += torch.count_nonzero(torch.argmax(Y_pred, dim=1) - Y) # 推定値と正解値が一致していないデータの個数を数える
    avg_loss = sum_loss / valid_size
    accuracy = (valid_size - n_failed) / valid_size
    print('valid loss = {0:.6f}'.format(avg_loss))
    print('accuracy = {0:.2f}%'.format(100 * accuracy))
    print('')

    # 学習途中のモデルの保存
    if AUTO_SAVE:
        torch.save(model.to('cpu').state_dict(), os.path.join(MODEL_DIR, 'autosaved_model_ep{0}.pth'.format(epoch + 1)))
        model.to(DEVICE)

    # 学習経過の可視化
    if VISUALIZE:
        visualizer.show(
            model,
            class_colors = [[255, 0, 0], [127, 127, 0], [0, 0, 255]],
            samples = samples_for_visualization,
            title = 'Epoch {0}'.format(epoch + 1)
        )

# 学習結果のニューラルネットワークモデルをファイルに保存
torch.save(model.to('cpu').state_dict(), MODEL_PATH)
