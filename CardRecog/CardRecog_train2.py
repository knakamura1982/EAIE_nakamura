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
from networks import CardClassifier2, CardChecker
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
CC_MODEL_PATH = os.path.join(MODEL_DIR, 'cc_model.pth')
PCC_MODEL_PATH = os.path.join(MODEL_DIR, 'pcc_model.pth')
NCC_MODEL_PATH = os.path.join(MODEL_DIR, 'ncc_model.pth')

# 画像のサイズ・チャンネル数
C = 3 # チャンネル数
H = 96 # 縦幅
W = 64 # 横幅


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
        'Suit', # Y1
        'Is Picture?', # Y2
        'Number (only picture card)', # Y3
        'Number (only non-picture card)', # Y4
    ],
    dtypes = [
        'image', # Xの型
        'label', # Y1の型
        'label', # Y2の型
        'label', # Y3の型
        'label', # Y4の型
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
        'Suit', # Y1
        'Is Picture?', # Y2
        'Number (only picture card)', # Y3
        'Number (only non-picture card)', # Y4
    ],
    dtypes = [
        'image', # Xの型
        'label', # Y1の型
        'label', # Y2の型
        'label', # Y3の型
        'label', # Y4の型
    ],
    dirname = DATA_DIR,
    fdicts = train_dataset.forward_dicts,
)
valid_size = len(valid_dataset)

# 訓練データおよび検証用データをミニバッチに分けて使用するための「データローダ」を用意
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ニューラルネットワークの作成
cc_model = CardChecker(C=C, H=H, W=W).to(DEVICE) # カードが絵札か否かを判定するモデル
pcc_model = CardClassifier2(C=C, H=H, W=W, C1=4, N_SUITS=4, N_NUMBERS=3).to(DEVICE) # 絵札カードのスートと数字を判定するモデル
ncc_model = CardClassifier2(C=C, H=H, W=W, C1=4, N_SUITS=4, N_NUMBERS=10).to(DEVICE) # 非絵札カードのスートと数字を判定するモデル

# 最適化アルゴリズムの指定（ここでは Adam を使用）
cc_optimizer = optim.Adam(cc_model.parameters())
pcc_optimizer = optim.Adam(pcc_model.parameters())
ncc_optimizer = optim.Adam(ncc_model.parameters())

# 損失関数：クロスエントロピー損失を使用
loss_func = nn.CrossEntropyLoss()
loss_viz = LossVisualizer(items=['train loss', 'valid loss'])

# 勾配降下法による繰り返し学習
for epoch in range(N_EPOCHS):

    print('Epoch {0}:'.format(epoch + 1))

    # 学習
    cc_model.train()
    pcc_model.train()
    ncc_model.train()
    sum_loss = 0
    for X, Y1, Y2, Y3, Y4 in tqdm(train_dataloader):
        for param in cc_model.parameters():
            param.grad = None
        for param in pcc_model.parameters():
            param.grad = None
        for param in ncc_model.parameters():
            param.grad = None
        X = X.to(DEVICE) # 画像（ミニバッチ）
        Y1 = Y1.to(DEVICE) # スートラベル（ミニバッチ）
        Y2 = Y2.to(DEVICE) # 絵札か否かの二値ラベル
        Y3 = Y3.to(DEVICE) # 絵札カードの数字ラベル
        Y4 = Y4.to(DEVICE) # 非絵札カードの数字ラベル
        p_idx = torch.where(Y2 == 1)[0] # ミニバッチ内の画像で実際に絵札であるもののインデックスを抽出
        n_idx = torch.where(Y2 == 0)[0] # ミニバッチ内の画像で実際に絵札でないもののインデックスを抽出
        Y2_pred = cc_model(X) # ミニバッチ内の各カードが絵札か否かを判定
        Y1_pred_p, Y3_pred = pcc_model(X[p_idx]) # 絵札カードのスートと数字を認識
        Y1_pred_n, Y4_pred = ncc_model(X[n_idx]) # 非絵札カードのスートと数字を認識

        # 損失関数の現在値を計算
        loss = loss_func(Y2_pred, Y2) + loss_func(Y1_pred_p, Y1[p_idx]) + loss_func(Y3_pred, Y3[p_idx]) + loss_func(Y1_pred_n, Y1[n_idx]) + loss_func(Y4_pred, Y4[n_idx])

        loss.backward() # 誤差逆伝播法により，個々のパラメータに関する損失関数の勾配（偏微分）を計算
        cc_optimizer.step() # 勾配に沿ってパラメータの値を更新
        pcc_optimizer.step() # 同上
        ncc_optimizer.step() # 同上
        sum_loss += float(loss) * len(X)
    avg_loss = sum_loss / train_size
    loss_viz.add_value('train loss', avg_loss)
    print('train loss = {0:.6f}'.format(avg_loss))

    # 検証
    cc_model.eval()
    pcc_model.eval()
    ncc_model.eval()
    sum_loss = 0
    n_failed = 0
    with torch.inference_mode():
        for X, Y1, Y2, Y3, Y4 in tqdm(valid_dataloader):
            X = X.to(DEVICE) # 画像（ミニバッチ）
            Y1 = Y1.to(DEVICE) # スートラベル（ミニバッチ）
            Y2 = Y2.to(DEVICE) # 絵札か否かの二値ラベル
            Y3 = Y3.to(DEVICE) # 絵札カードの数字ラベル
            Y4 = Y4.to(DEVICE) # 非絵札カードの数字ラベル
            Y2_pred = cc_model(X).detach() # ミニバッチ内の各カードが絵札か否かを判定
            p_idx = torch.where(Y2 == 1)[0] # ミニバッチ内の画像で実際に絵札であるもののインデックスを抽出
            n_idx = torch.where(Y2 == 0)[0] # ミニバッチ内の画像で実際に絵札でないもののインデックスを抽出
            Y1_pred_p, Y3_pred = pcc_model(X[p_idx]) # 絵札カードのスートと数字を認識
            Y1_pred_n, Y4_pred = ncc_model(X[n_idx]) # 非絵札カードのスートと数字を認識

            # 損失関数の現在値を計算
            loss = loss_func(Y2_pred, Y2) + loss_func(Y1_pred_p, Y1[p_idx]) + loss_func(Y3_pred, Y3[p_idx]) + loss_func(Y1_pred_n, Y1[n_idx]) + loss_func(Y4_pred, Y4[n_idx])
            sum_loss += float(loss) * len(X)

            # 推定値と正解値が一致していないデータの個数を数える
            n_failed += torch.count_nonzero(
                torch.abs(torch.argmax(Y2_pred[p_idx], dim=1) - Y2[p_idx])
                + torch.abs(torch.argmax(Y1_pred_p, dim=1) - Y1[p_idx])
                + torch.abs(torch.argmax(Y3_pred, dim=1) - Y3[p_idx])
            ) + torch.count_nonzero(
                torch.abs(torch.argmax(Y2_pred[n_idx], dim=1) - Y2[n_idx])
                + torch.abs(torch.argmax(Y1_pred_n, dim=1) - Y1[n_idx])
                + torch.abs(torch.argmax(Y4_pred, dim=1) - Y4[n_idx])
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
        torch.save(cc_model.to('cpu').state_dict(), os.path.join(MODEL_DIR, 'autosaved_cc_model_ep{0}.pth'.format(epoch + 1)))
        torch.save(pcc_model.to('cpu').state_dict(), os.path.join(MODEL_DIR, 'autosaved_pcc_model_ep{0}.pth'.format(epoch + 1)))
        torch.save(ncc_model.to('cpu').state_dict(), os.path.join(MODEL_DIR, 'autosaved_ncc_model_ep{0}.pth'.format(epoch + 1)))
        cc_model.to(DEVICE)
        pcc_model.to(DEVICE)
        ncc_model.to(DEVICE)

# 学習結果のニューラルネットワークモデルをファイルに保存
torch.save(cc_model.to('cpu').state_dict(), CC_MODEL_PATH)
torch.save(pcc_model.to('cpu').state_dict(), PCC_MODEL_PATH)
torch.save(ncc_model.to('cpu').state_dict(), NCC_MODEL_PATH)
