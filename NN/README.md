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
