import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
sys.path.append(os.path.abspath('..'))
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from networks import SampleMLP
from mylib.data_io import CSVBasedDataset
from mylib.option import print_args


# データセットファイル
DATASET_CSV = './csv_data/weather_test.csv'

# 学習結果の保存先フォルダ
MODEL_DIR = './MLP_models'


# デバイス, バッチサイズなどをコマンドライン引数から取得し変数に保存
parser = argparse.ArgumentParser(description='Multi-Layer Perceptron Sample Code (test)')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU/CUDA ID (negative value indicates CPU)')
parser.add_argument('--batchsize', '-b', default=100, type=int, help='minibatch size')
parser.add_argument('--model', '-m', default=os.path.join(MODEL_DIR, 'model.pth'), type=str, help='file path of trained model')
args = print_args(parser.parse_args())
DEVICE = args['device']
BATCH_SIZE = args['batchsize']
MODEL_PATH = args['model']

# CSVファイルを読み込み, テストデータセットを用意
test_dataset = CSVBasedDataset(
    filename = DATASET_CSV,
    items = [
        ['平均気温', '平均湿度'], # X
        '天気概況' # Y
    ],
    dtypes = [
        'float', # Xの型
        'label' # Yの型
    ]
)
test_size = len(test_dataset)

# テストデータをミニバッチに分けて使用するための「データローダ」を用意
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ニューラルネットワークの作成
model = SampleMLP()
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)
model.eval()

# テストデータセットを用いて認識精度を評価
n_failed = 0
with torch.inference_mode():
    for X, Y in tqdm(test_dataloader):
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)
        Y_pred = model(X)
        n_failed += torch.count_nonzero(torch.argmax(Y_pred, dim=1) - Y) # 推定値と正解値が一致していないデータの個数を数える
accuracy = (test_size - n_failed) / test_size
print('accuracy = {0:.2f}%'.format(100 * accuracy))
print('')
