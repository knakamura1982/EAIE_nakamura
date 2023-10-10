import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
sys.path.append(os.path.abspath('..'))
import pickle
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from networks import SampleCNN, myCNN
from mylib.data_io import CSVBasedDataset
from mylib.utility import print_args


# データセットファイル
DATASET_CSV = './MNIST/test_list.csv'

# 画像ファイル名の先頭に付加する文字列（画像ファイルが存在するディレクトリのパス）
DATA_DIR = './MNIST/'

# 学習結果の保存先フォルダ
MODEL_DIR = './CNN_models'

# 画像のサイズ・チャンネル数
C = 1 # チャンネル数
H = 28 # 縦幅
W = 28 # 横幅


# デバイス, バッチサイズなどをコマンドライン引数から取得し変数に保存
parser = argparse.ArgumentParser(description='Convolutional Neural Network Sample Code (test)')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU/CUDA ID (negative value indicates CPU)')
parser.add_argument('--batchsize', '-b', default=100, type=int, help='minibatch size')
parser.add_argument('--model', '-m', default=os.path.join(MODEL_DIR, 'model.pth'), type=str, help='file path of trained model')
args = print_args(parser.parse_args())
DEVICE = args['device']
BATCH_SIZE = args['batchsize']
MODEL_PATH = args['model']


# CSVファイルを読み込み, 訓練データセットを用意
with open(os.path.join(MODEL_DIR, 'fdicts.pkl'), 'rb') as fdicts_file:
    fdicts = pickle.load(fdicts_file)
test_dataset = CSVBasedDataset(
    filename = DATASET_CSV,
    items = [
        'File Path', # X
        'Class Label' # Y
    ],
    dtypes = [
        'image', # Xの型
        'label' # Yの型
    ],
    dirname = DATA_DIR,
    fdicts = fdicts,
)
test_size = len(test_dataset)

# 認識対象のクラス数を取得
n_classes = len(test_dataset.forward_dicts[1])

# テストデータをミニバッチに分けて使用するための「データローダ」を用意
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ニューラルネットワークの作成
model = SampleCNN(C=C, H=H, W=W, N=n_classes)
#model = myCNN(C=C, H=H, W=W, N=n_classes) # myCNNクラスを用いる場合はこちらを使用
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
