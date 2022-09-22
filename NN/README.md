### csv_data

サンプルデータセット（CSVファイル）が保存されているフォルダ．

### networks.py

使用するニューラルネットワークのネットワーク構造が記載されているファイル．

### MLP_train.py

**コマンド**
```
python MLP_train.py --gpu 0 --epochs 50 --batchsize 100 --model MLP_model.pth --autosave
```
**オプション**
- gpu
  - 使用するGPUのID (-1を指定するとCPU上で動作する)
  - このオプションを指定しない場合，デフォルト値として -1 がセットされる
  - cudaを使用できない環境では無視される
- epochs
  - 何エポック分学習するか
  - このオプションを指定しない場合，デフォルト値として 50 がセットされる
- batchsize
  - バッチサイズ
  - このオプションを指定しない場合，デフォルト値として 100 がセットされる
- model
  - 学習結果のモデルパラメータの保存先
  - このオプションを指定しない場合，デフォルト値として ./MLP_models/model.pth がセットされる
- autosave
  - 指定すると毎エポック終了時にモデルパラメータが自動保存される
  - 保存先は ./MLP_models/autosaved_model_epX.pth となる（ X はエポック番号 ）

### MLP_test.py

**コマンド**
```
python MLP_test.py --gpu 0 --batchsize 100 --model MLP_model.pth
```
**オプション**
- gpu
  - 使用するGPUのID
  - デフォルト値も含めて MLP_train.py の同名オプションと同じ
- batchsize
  - バッチサイズ（テスト時でもデータ数が多い場合にはミニバッチ分割しないと out of memory になる）
  - デフォルト値も含めて MLP_train.py の同名オプションと同じ
- model
  - ロードするモデルパラメータファイル
  - このオプションを指定しない場合，デフォルトとして ./MLP_models/model.pth が読み込まれる

### MLP_models

MLP_train.py による学習結果の保存先として使用する想定のフォルダ．  
動作テスト時に作成したファイルが model.pth という名前で残っているが，決して良いモデルではない

### MNIST.tar.gz

手書き数字文字画像データセットMNIST（下記URL参照）を一括圧縮したファイル．  
のちに画像認識について説明する機会が訪れたら使用する想定でいるが，当面は使用しない（これは CNN_train.py, CNN_test.py も同様）
http://yann.lecun.com/exdb/mnist/

ターミナル，もしくはコマンドプロンプト（Windows 10以上）で以下のコマンドを打つことにより解凍できる
```
tar -xzvf MNIST.tar.gz
```

### CNN_train.py

### CNN_test.py

### CNN_models

CNN_train.py による学習結果の保存先として使用する想定のフォルダ．  
動作テスト時に作成したファイルが model.pth という名前で残っているが，決して良いモデルではない
