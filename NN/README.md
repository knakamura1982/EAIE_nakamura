## csv_data

サンプルデータセット（CSVファイル）が保存されているフォルダです．

## networks.py

使用するニューラルネットワークのネットワーク構造が記載されているファイルです．

## MLP_train.py

CSVの学習データセットを読み込んでニューラルネットワーク（多層パーセプトロン）を学習するプログラム．  
読み込むデータセットは18行目で指定します．

25行目の VISUALIZE = True は学習プロセスを可視化するための設定で，  
データセットとして weather_train.csv を使用することを前提としています．  
可視化が不要な場合や，データセットを weather_train.csv から変更する場合は，VISUALIZE = False としてください．

**コマンド**
```
python MLP_train.py --gpu 0 --epochs 50 --batchsize 100 --model MLP_model.pth --autosave
```
**オプション**
- gpu
  - 使用するGPUのID (-1を指定するとCPU上で動作します)
  - このオプションを指定しない場合，デフォルト値として -1 がセットされます．
  - cudaを使用できない環境では無視されます．
- epochs
  - 何エポック分学習するか
  - このオプションを指定しない場合，デフォルト値として 50 がセットされます．
- batchsize
  - バッチサイズ
  - このオプションを指定しない場合，デフォルト値として 100 がセットされます．
- model
  - 学習結果のモデルパラメータの保存先
  - このオプションを指定しない場合，デフォルト値として ./MLP_models/model.pth がセットされます．
- autosave
  - 指定すると毎エポック終了時にモデルパラメータが自動保存されるようになります．
  - 保存先は ./MLP_models/autosaved_model_epX.pth です（ X はエポック番号 ）．

## MLP_test.py

上記の MLP_train.py で学習したニューラルネットワークのテストを行うプログラム．  
使用するテストデータセットは15行目で指定します．

**コマンド**
```
python MLP_test.py --gpu 0 --batchsize 100 --model MLP_model.pth
```
**オプション**
- gpu
  - 使用するGPUのID
  - デフォルト値も含めて MLP_train.py の同名オプションと同じです．
- batchsize
  - バッチサイズ（テスト時でもデータ数が多い場合にはミニバッチ分割しないと out of memory になるため，このオプションを用意しました）
  - デフォルト値も含めて MLP_train.py の同名オプションと同じです．
- model
  - ロードするモデルパラメータファイル
  - このオプションを指定しない場合，デフォルトとして ./MLP_models/model.pth が読み込まれます．

## MLP_models

MLP_train.py による学習結果の保存先として使用する想定のフォルダ．  
動作テスト時に作成したファイルが model.pth という名前で残っていますが，決して良いモデルではありません．

## MNIST.tar.gz

手書き数字文字画像データセットMNIST（下記URL参照）を一括圧縮したファイル．  
のちに画像認識について説明する機会が訪れたら使用する想定でいますが，  
当面は使用しません（これは CNN_train.py, CNN_test.py も同様）．  
http://yann.lecun.com/exdb/mnist/

ターミナル，もしくはコマンドプロンプト（Windows 10以上）で以下のコマンドを打つことにより解凍できます．
```
tar -xzvf MNIST.tar.gz
```

## CNN_train.py

画像データセットを読み込んで畳込みニューラルネットワークを学習するプログラム．  
読み込むデータセットは17行目で指定します．

**コマンド**
```
python CNN_train.py --gpu 0 --epochs 10 --batchsize 100 --model CNN_model.pth --autosave
```
**オプション**
- gpu
  - 使用するGPUのID
  - デフォルト値も含めて MLP_train.py の同名オプションと同じです．
- epochs
  - 何エポック分学習するか
  - MLP_train.py の同名オプションと同じですが，指定しない場合のデフォルト値は 10 となっています．
- batchsize
  - バッチサイズ
  - デフォルト値も含めて MLP_train.py の同名オプションと同じです．
- model
  - 学習結果のモデルパラメータの保存先
  - このオプションを指定しない場合，デフォルト値として ./CNN_models/model.pth がセットされます．
- autosave
  - 指定すると毎エポック終了時にモデルパラメータが自動保存されるようになります．
  - 保存先は ./CNN_models/autosaved_model_epX.pth です（ X はエポック番号 ）．

## CNN_test.py

上記の CNN_train.py で学習したニューラルネットワークのテストを行うプログラム．  
使用するテストデータセットは15行目で指定します．

**コマンド**
```
python CNN_test.py --gpu 0 --batchsize 100 --model CNN_model.pth
```
**オプション**
- gpu
  - 使用するGPUのID
  - デフォルト値も含めて MLP_train.py の同名オプションと同じです．
- batchsize
  - バッチサイズ
  - デフォルト値も含めて MLP_train.py の同名オプションと同じです．
- model
  - ロードするモデルパラメータファイル
  - このオプションを指定しない場合，デフォルトとして ./CNN_models/model.pth が読み込まれます．

## CNN_models

CNN_train.py による学習結果の保存先として使用する想定のフォルダ．  
動作テスト時に作成したファイルが model.pth という名前で残っていますが，決して良いモデルではありません．
