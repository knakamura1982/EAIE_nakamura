import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
sys.path.append(os.path.abspath('..'))
import cv2
import pickle
import argparse
import torch
import numpy as np
from square_detecters import SquareDetector
from networks import CardClassifierAlt, CardChecker
from mylib.utility import print_args


# 学習結果の保存先フォルダ
MODEL_DIR = './CNN_models'
CC_MODEL_PATH = os.path.join(MODEL_DIR, 'cc_model.pth')
PCC_MODEL_PATH = os.path.join(MODEL_DIR, 'pcc_model.pth')
NCC_MODEL_PATH = os.path.join(MODEL_DIR, 'ncc_model.pth')

# カード画像のサイズ・チャンネル数
C = 3 # チャンネル数
H = 144 # 縦幅
W = 96  # 横幅

# カード画像のサイズ(Height, Width)
CARD_IMAGE_SIZE = (H, W)

# 接続される可能性のあるデバイスの総数（適当）
MAX_DEVICE_ID = 10

# 四角形検出結果を表示する際の検出枠の太さ
RECT_THINKNESS = 3

# ウインドウ設定
WINDOW_NAME = 'camera frame'
KEY_INTERVAL = 30


# 指定された四角形領域に対しホモグラフィ変換を施し長方形化する
#   - img: 入力画像
#   - square: 変換対象の四角形領域（SquareDetector.detect()メソッドが出力した領域リストの要素）
#   - out_size: ホモグラフィ変換後の出力画像のサイズ(Height, Width)
def homography(img:np.ndarray, square:np.ndarray, out_size:tuple):

    h, w = out_size

    # x座標とy座標の和が最も小さい頂点を左上隅とみなす
    min_pos = square[0, 0, 0] + square[0, 0, 1]
    base_index = 0
    for i in range(1, 4):
        pos = square[i, 0, 0] + square[i, 0, 1]
        if min_pos > pos:
            min_pos = pos
            base_index = i
    square = np.roll(square, -base_index, axis=0)

    # 各頂点が時計回りに番号づけられるように調整
    if (square[1,0,0] - square[0,0,0])*(square[-1,0,1] - square[0,0,1]) - (square[1,0,1] - square[0,0,1])*(square[-1,0,0] - square[0,0,0]) < 0:
        source_points = np.array([square[0, 0], square[3, 0], square[2, 0], square[1, 0]], dtype=np.float32)
    else:
        source_points = np.array([square[0, 0], square[1, 0], square[2, 0], square[3, 0]], dtype=np.float32)
    target_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # ホモグラフィ行列を計算・適用
    affine_mat = cv2.getPerspectiveTransform(source_points, target_points)
    ret = cv2.warpPerspective(img, affine_mat, (w, h), flags=cv2.INTER_AREA)

    return ret


# マウス処理（マウスコールバック関数）
def mouse_events(event, x, y, flags, param):

    global current_camera_id, camera_num

    try:

        # 左クリック時
        if event == cv2.EVENT_LBUTTONDOWN:
            pass

        # 右クリック時
        if event == cv2.EVENT_RBUTTONDOWN:
            current_camera_id = (current_camera_id + 1) % camera_num
            print('switch to camera {0}'.format(current_camera_id))

    except Exception as e:
        print(e)


# デバイス, バッチサイズなどをコマンドライン引数から取得し変数に保存
parser = argparse.ArgumentParser(description='Convolutional Neural Network Sample Code (test)')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU/CUDA ID (negative value indicates CPU)')
args = print_args(parser.parse_args())
DEVICE = args['device']

# 認識対象クラスの名称と番号の対応表を取得
with open(os.path.join(MODEL_DIR, 'fdicts2.pkl'), 'rb') as fdicts_file:
    fdicts = pickle.load(fdicts_file)
    fdict1 = fdicts[1]
    fdict3 = fdicts[3]
    fdict4 = fdicts[4]
    rdict1 = {v:k for k, v in fdict1.items()}
    rdict3 = {v:k for k, v in fdict3.items()}
    rdict4 = {v:k for k, v in fdict4.items()}

# ニューラルネットワークの作成
cc_model = CardChecker(C=C, H=H, W=W) # カードが絵札か否かを判定するモデル
pcc_model = CardClassifierAlt(C=C, H=H, W=W, N_SUITS=4, N_NUMBERS=3) # 絵札カードのスートと数字を判定するモデル
ncc_model = CardClassifierAlt(C=C, H=H, W=W, N_SUITS=4, N_NUMBERS=10) # 非絵札カードのスートと数字を判定するモデル
cc_model.load_state_dict(torch.load(CC_MODEL_PATH))
pcc_model.load_state_dict(torch.load(PCC_MODEL_PATH))
ncc_model.load_state_dict(torch.load(NCC_MODEL_PATH))
cc_model = cc_model.to(DEVICE)
pcc_model = pcc_model.to(DEVICE)
ncc_model = ncc_model.to(DEVICE)
cc_model.eval()
pcc_model.eval()
ncc_model.eval()

# カメラを駆動
captures = []
for camera_device_id in range(MAX_DEVICE_ID):
    cap = cv2.VideoCapture(camera_device_id)
    if cap.isOpened():
        captures.append(cap)
camera_num = len(captures)
print('num. of detected cameras: ', camera_num)

# 四角形検出器を用意
square_detector = SquareDetector()

# 表示用ウインドウを作成
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, mouse_events)

# 撮影開始
current_camera_id = camera_num - 1
while True:

    # 現フレームの画像を取得
    ret, frame = captures[current_camera_id].read()
    if ret:

        # 四角形を検出
        square = square_detector.detect(frame, multi_squares=False)

        # 検出に成功した場合は...
        if len(square) > 0:

            # 検出領域を補正・リサイズ
            img = homography(frame, square[0], out_size=CARD_IMAGE_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

            # カードの種類を認識
            x = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(dim=0).to(DEVICE)
            y2 = int(torch.argmax(cc_model(x), dim=1).to('cpu').detach()[0]) # x が絵札か否かを判定
            if y2 == 0:
                # 非絵札の場合
                y1, y4 = ncc_model(x)
                y1 = int(torch.argmax(y1, dim=1).to('cpu').detach()[0])
                y4 = int(torch.argmax(y4, dim=1).to('cpu').detach()[0])
                class_name = rdict1[y1] + '-' + rdict4[y4]
            else:
                # 絵札の場合
                y1, y3 = pcc_model(x)
                y1 = int(torch.argmax(y1, dim=1).to('cpu').detach()[0])
                y3 = int(torch.argmax(y3, dim=1).to('cpu').detach()[0])
                class_name = rdict1[y1] + '-' + rdict3[y3]

            # カードの種類（認識結果）を描画
            p = np.min(square[0], axis=(0, 1))
            cv2.putText(frame, text=class_name, org=(p[0]-30, p[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 192, 64), thickness=2, lineType=cv2.LINE_4)

        # 検出領域を元画像に描画
        square_detector.draw(frame, square, color=(0, 192, 64), thickness=RECT_THINKNESS)

        # 元画像を表示
        cv2.imshow(WINDOW_NAME, frame)

    # ESCキーが押下されたら終了
    key = cv2.waitKey(KEY_INTERVAL)
    if key == 27:
        cv2.destroyWindow(WINDOW_NAME) # 表示用ウインドウを破棄
        break

    # ウインドウの「×」ボタンが押された場合も終了
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) == 0:
        break

# カメラをリリース
for cap in captures:
    cap.release()
