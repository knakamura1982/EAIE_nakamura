import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor


# 活性化関数の設定モジュール
# 引数 activation が str 型のときは以下の通り
#   - 'n': 活性化関数なし
#   - 's': sigmoid
#   - 't': tanh
#   - 'r': ReLU
#   - 'l': Leaky ReLU
#   - 'e': ELU
# 引数 activation が関数そのもののときは，それがそのまま設定される
def __select_activation__(activation):
    if type(activation) is str:
        activation = activation.lower()
        if activation == 'n' or activation == 'none':
            return None
        elif activation == 's' or activation == 'sigmoid':
            return torch.sigmoid
        elif activation == 't' or activation == 'tanh':
            return torch.tanh
        elif activation == 'r' or activation == 'relu':
            return F.relu
        elif activation == 'l' or activation == 'leaky_relu':
            return F.leaky_relu
        elif activation == 'e' or activation == 'elu':
            return F.elu
    else:
        return activation


# 畳込み + バッチ正規化 + 活性化関数を行う層
# kernel_size を奇数に設定した上で stride と padding を指定しなければ（デフォルト値にすれば）, 出力マップと入力マップは同じサイズになる
#   - in_channels: 入力マップのチャンネル数
#   - out_channels: 出力マップのチャンネル数
#   - kernel_size: カーネルサイズ（デフォルトで 3 になる）
#   - stride: ストライド幅（デフォルトで 1 になる）
#   - padding: パディングサイズ（デフォルトで (kernel_size-1)/2 になる）
#   - do_bn: バッチ正規化を行うか否か（Trueなら行う, Falseなら行わない. デフォルトでは行う）
#   - dropout_ratio: 0 以外ならその割合でドロップアウト処理を実行（デフォルトでは 0, すなわちドロップアウトなし）
#   - activation: 活性化関数（デフォルトでは ReLU になる）
class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=-1, do_bn=True, dropout_ratio=0, activation=F.relu):
        super(Conv, self).__init__()
        if type(padding) == int and padding < 0:
            padding = (kernel_size - 1) // 2
        self.do_bn = do_bn
        self.dropout_ratio = dropout_ratio
        self.activation = __select_activation__(activation)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        if do_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels)
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)

    def __call__(self, x):
        h = self.conv(x) # 畳込み
        if self.do_bn:
            h = self.bn(h) # バッチ正規化
        if not self.activation is None:
            h = self.activation(h) # 活性化関数
        if self.dropout_ratio != 0:
            h = self.dropout(h) # ドロップアウト
        return h


# 畳込み + バッチ正規化 + 活性化関数を行う層
# 上記 Conv のラッパークラスで, 出力マップのサイズが入力マップの半分になるように stride と padding が自動設定される
# 入力マップのサイズは縦横ともに偶数であり, kernel_size にも偶数が指定されることを前提としている（そうでない場合の動作は保証外）
# なお, kernel_size のデフォルト値は 4
class ConvHalf(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, do_bn=True, dropout_ratio=0, activation=F.relu):
        super(ConvHalf, self).__init__()
        self.conv = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=(kernel_size-1)//2,
            do_bn=do_bn,
            dropout_ratio=dropout_ratio,
            activation=activation)

    def __call__(self, x):
        return self.conv(x)


# 逆畳込み + バッチ正規化 + 活性化関数を行う層
# kernel_size を偶数に設定した上で stride と padding を指定しなければ（デフォルト値にすれば）, 出力マップのサイズは入力マップのちょうど2倍になる
#   - in_channels: 入力マップのチャンネル数
#   - out_channels: 出力マップのチャンネル数
#   - kernel_size: カーネルサイズ（デフォルトで 4 になる）
#   - stride: ストライド幅（デフォルトで 2 になる）
#   - padding: パディングサイズ（デフォルトで (kernel_size-1)/2 になる）
#   - do_bn: バッチ正規化を行うか否か（Trueなら行う, Falseなら行わない. デフォルトでは行う）
#   - dropout_ratio: 0 以外ならその割合でドロップアウト処理を実行（デフォルトでは 0, すなわちドロップアウトなし）
#   - activation: 活性化関数（デフォルトでは ReLU になる）
class Deconv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=-1, do_bn=True, dropout_ratio=0, activation=F.relu):
        super(Deconv, self).__init__()
        if type(padding) == int and padding < 0:
            padding = (kernel_size - 1) // 2
        self.do_bn = do_bn
        self.dropout_ratio = dropout_ratio
        self.activation = __select_activation__(activation)
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        if do_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels)
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)

    def __call__(self, x):
        h = self.deconv(x) # 逆畳込み
        if self.do_bn:
            h = self.bn(h) # バッチ正規化
        if not self.activation is None:
            h = self.activation(h) # 活性化関数
        if self.dropout_ratio != 0:
            h = self.dropout(h) # ドロップアウト
        return h


# プーリングを行う層
#   - method: プーリング手法（'max'または'avg'のいずれか, デフォルトでは average pooling ）
#   - scale: カーネルサイズおよびストライド幅
class Pool(nn.Module):

    def __init__(self, method='avg', scale=2):
        super(Pool, self).__init__()
        if method == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=scale, stride=scale)
        elif method == 'max':
            self.pool = nn.MaxPool2d(kernel_size=scale, stride=scale)

    def __call__(self, x):
        return self.pool(x)


# Global Pooling を行う層
#   - method: プーリング手法（'max'または'avg'のいずれか, デフォルトでは global average pooling ）
class GlobalPool(nn.Module):

    def __init__(self, method='avg'):
        super(GlobalPool, self).__init__()
        self.method = method

    def __call__(self, x):
        if self.method == 'avg':
            h = torch.mean(x, dim=(2, 3))
        elif self.method == 'max':
            h = torch.max(x, dim=3)[0]
            h = torch.max(h, dim=2)[0]
        return h


# 畳込み + バッチ正規化 + 活性化関数 + Pixel Shuffle を行う層
# 畳込みにおける stride と padding は, 出力マップのサイズが入力マップのサイズの scale 倍になるように自動設定する
#   - in_channels: 入力マップのチャンネル数
#   - out_channels: 出力マップのチャンネル数
#   - scale: Pixel Shuffle におけるスケールファクタ（デフォルトで 2 になる）
#   - kernel_size: カーネルサイズ（奇数のみ可, 偶数の場合の動作は保証外. デフォルトで 3 になる）
#   - do_bn: バッチ正規化を行うか否か（Trueなら行う, Falseなら行わない. デフォルトでは行う）
#   - dropout_ratio: 0 以外ならその割合でドロップアウト処理を実行（デフォルトでは 0, すなわちドロップアウトなし）
#   - activation: 活性化関数（デフォルトでは ReLU になる）
class PixShuffle(nn.Module):

    def __init__(self, in_channels, out_channels, scale=2, kernel_size=3, do_bn=True, dropout_ratio=0, activation=F.relu):
        super(PixShuffle, self).__init__()
        self.do_bn = do_bn
        self.dropout_ratio = dropout_ratio
        self.activation = __select_activation__(activation)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels*scale*scale, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.ps = nn.PixelShuffle(scale)
        if do_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels*scale*scale)
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)

    def __call__(self, x):
        h = self.conv(x) # 畳込み
        if self.do_bn:
            h = self.bn(h) # バッチ正規化
        if not self.activation is None:
            h = self.activation(h) # 活性化関数
        h = self.ps(h) # pixel shuffle
        if self.dropout_ratio != 0:
            h = self.dropout(h) # ドロップアウト
        return h


# 全結合 + バッチ正規化 + 活性化関数を行う層
#   - in_features: 入力のパーセプトロン数
#   - out_features: 出力のパーセプトロン数
#   - do_bn: バッチ正規化を行うか否か（Trueなら行う, Falseなら行わない. デフォルトでは行う）
#   - dropout_ratio: 0 以外ならその割合でドロップアウト処理を実行（デフォルトでは 0, すなわちドロップアウトなし）
#   - activation: 活性化関数（デフォルトでは ReLU になる）
class FC(nn.Module):

    def __init__(self, in_features, out_features, do_bn=True, dropout_ratio=0, activation=F.relu):
        super(FC, self).__init__()
        self.do_bn = do_bn
        self.dropout_ratio = dropout_ratio
        self.activation = __select_activation__(activation)
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        if do_bn:
            self.bn = nn.BatchNorm1d(num_features=out_features)
        if dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)

    def __call__(self, x):
        h = self.fc(x) # 全結合
        if self.do_bn:
            h = self.bn(h) # バッチ正規化
        if not self.activation is None:
            h = self.activation(h) # 活性化関数
        if self.dropout_ratio != 0:
            h = self.dropout(h) # ドロップアウト
        return h


# 平坦化を行う層
class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()
        self.flat = nn.Flatten()

    def __call__(self, x):
        return self.flat(x)


# 特徴マップの再配置（平坦化の逆）を行う層
#   - size: 再配置後の特徴マップのサイズ
#     -- size[0]: チャンネル数
#     -- size[1]: 縦幅
#     -- size[2]: 横幅
class Reshape(nn.Module):

    def __init__(self, size):
        super(Reshape, self).__init__()
        self.size = size

    def __call__(self, x):
        batchsize = x.size()[0]
        h = x.reshape((batchsize, *self.size))
        return h


# Adaptive Instance Normalization
#   - x: コンテンツ特徴マップ
#   - y: スタイル特徴マップ
class MapAdaIN(nn.Module):

    def __init__(self, in_channels_content, in_channels_style, do_conv=True):
        super(MapAdaIN, self).__init__()
        self.do_conv = do_conv
        if do_conv:
            self.conv_s = Conv(in_channels=in_channels_style, out_channels=in_channels_content, kernel_size=1, stride=1, padding=0, do_bn=False, activation='none')
            self.conv_c = Conv(in_channels=in_channels_content, out_channels=in_channels_content, kernel_size=1, stride=1, padding=0, do_bn=False, activation='none')

    def __call__(self, x, y):
        if self.do_conv:
            x = self.conv_c(x)
            y = self.conv_s(y)
        xu = torch.mean(x, dim=(2, 3), keepdim=True) # コンテンツ特徴マップのチャンネルごとの平均
        yu = torch.mean(y, dim=(2, 3), keepdim=True) # スタイル特徴マップのチャンネルごとの平均
        xs = torch.std(x, dim=(2, 3), unbiased=False, keepdim=True) # コンテンツ特徴マップのチャンネルごとの標準偏差
        ys = torch.std(y, dim=(2, 3), unbiased=False, keepdim=True) # スタイル特徴マップのチャンネルごとの標準偏差
        h = ys * ((x - xu) / xs) + yu
        return h


# Adaptive Instance Normalization
#   - x: コンテンツ特徴マップ
#   - y: スタイル特徴ベクトル
class VecAdaIN(nn.Module):

    def __init__(self, in_channels_content, in_features_style, do_conv=True):
        super(VecAdaIN, self).__init__()
        self.do_conv = do_conv
        if do_conv:
            self.conv = Conv(in_channels=in_channels_content, out_channels=in_channels_content, kernel_size=1, stride=1, padding=0, do_bn=False, activation='none')
        self.mu = FC(in_features=in_features_style, out_features=in_channels_content, do_bn=False, activation=None)
        self.lnvar = FC(in_features=in_features_style, out_features=in_channels_content, do_bn=False, activation=None)

    def __call__(self, x, y):
        if self.do_conv:
            x = self.conv(x)
        xu = torch.mean(x, dim=(2, 3), keepdim=True)
        xs = torch.std(x, dim=(2, 3), unbiased=False, keepdim=True)
        yu = self.mu(y)
        ys = torch.exp(0.5 * self.lnvar(y))
        h = ys.reshape((*ys.size(), 1, 1)) * ((x - xu) / xs) + yu.reshape((*yu.size(), 1, 1))
        return h


# 転移学習で VGG や ResNet を使用する際の前処理を行う層
class BackbonePreprocess(nn.Module):

    def __init__(self, do_center_crop=True):
        super(BackbonePreprocess, self).__init__()
        if do_center_crop:
            self.preprocess = transforms.Compose([
                transforms.Resize(256), # 入力画像を 256x256 ピクセルに正規化
                transforms.CenterCrop(224), # 256x256 ピクセルの入力画像から中央 224x224 ピクセルを取り出す
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 取り出した部分の画素値を正規化する
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize(224), # 入力画像を 224x224 ピクセルに正規化
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 取り出した部分の画素値を正規化する
            ])

    def __call__(self, x):
        return self.preprocess(x)


# 転移学習でバックボーンモデルとして使用する層
#   - model: バックボーンとして使用する有名モデル（例えば models.resnet18(pretrained=True) など）
#   - layer_name: 特徴量として使用する層の名称（この層より出力側の層は削除される）
#   - finetune: 単に転移するだけか, ファインチューニングを行うか（True だとファインチューニング, False だと転移学習のみ. デフォルトでは False ）
class Backbone(nn.Module):

    def __init__(self, model, layer_name, finetune=False):
        super(Backbone, self).__init__()
        self.finetune = finetune
        self.backbone = create_feature_extractor(model, {layer_name: 'feature'})
        if not finetune:
            # バックボーンモデルのパラメータを固定
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def __call__(self, x):
        return self.backbone(x)['feature']

    def train(self):
        super().train()
        if not self.finetune:
            self.backbone.eval()

    def print_output_size(self):
        for param in self.parameters():
            device = param.data.device
            break
        x = torch.randn(1, 3, 224, 224).to(device)
        y = self.__call__(x)
        print(y.size()[1:])


# Plain 型 Residual Block 層
# 畳込みにおける stride と padding は, 出力マップが入力マップと同じサイズになるように自動決定する
#   - in_channels: 入力マップのチャンネル数
#   - out_channels: 出力マップのチャンネル数（0 以下の時は自動的に in_channels と同じ値を設定，デフォルトでは 0 ）
#   - kernel_size: カーネルサイズ（奇数のみ可, 偶数の場合の動作は保証外. デフォルトで 3 になる）
#   - do_bn: バッチ正規化を行うか否か（Trueなら行う, Falseなら行わない. デフォルトでは行う）
#   - dropout_ratio: 0 以外ならその割合でドロップアウト処理を実行（デフォルトでは 0, すなわちドロップアウトなし）
#   - activation: 活性化関数（デフォルトでは ReLU になる）
class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels=0, kernel_size=3, do_bn=True, dropout_ratio=0, activation=F.relu):
        super(ResBlock, self).__init__()
        self.do_bn = do_bn
        self.dropout_ratio = dropout_ratio
        self.activation = __select_activation__(activation)
        if out_channels <= 0:
            out_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        if self.do_bn:
            self.bn1 = nn.BatchNorm2d(num_features=in_channels)
            self.bn2 = nn.BatchNorm2d(num_features=in_channels)
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)
        if in_channels != out_channels:
            self.use_resconv = True
            self.resconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            if self.do_bn:
                self.resbn = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.use_resconv = False

    def __call__(self, x):
        if self.do_bn:
            h = self.activation(self.bn1(self.conv1(x))) # 畳込み＆バッチ正規化＆活性化関数
            h = self.bn2(self.conv2(h)) # 畳込み＆バッチ正規化
        else:
            h = self.activation(self.conv1(x)) # 畳込み＆活性化関数
            h = self.conv2(h) # 畳込み
        h = self.activation(h + x) # 加算＆活性化関数
        if self.use_resconv:
            if self.do_bn:
                h = self.activation(self.resbn(self.resconv(h))) # チャンネル数を変更するための畳込み＆バッチ正規化
            else:
                h = self.activation(self.resconv(h)) # チャンネル数を変更するための畳込み
        if self.dropout_ratio != 0:
            h = self.dropout(h) # ドロップアウト
        return h


# Bottle-Neck 型 Residual Block 層
# 畳込みにおける stride と padding は, 出力マップが入力マップと同じサイズになるように自動決定する
#   - in_channels: 入力マップのチャンネル数
#   - out_channels: 出力マップのチャンネル数（0 以下の時は自動的に in_channels と同じ値を設定，デフォルトでは 0 ）
#   - mid_channels: 中間層のマップのチャンネル数（0 以下の時は自動的に in_channels // 4 を設定, デフォルトでは 0 ）
#   - kernel_size: カーネルサイズ（奇数のみ可, 偶数の場合の動作は保証外. デフォルトで 3 になる）
#   - do_bn: バッチ正規化を行うか否か（Trueなら行う, Falseなら行わない. デフォルトでは行う）
#   - dropout_ratio: 0 以外ならその割合でドロップアウト処理を実行（デフォルトでは 0, すなわちドロップアウトなし）
#   - activation: 活性化関数（デフォルトでは ReLU になる）
class ResBlockBN(nn.Module):

    def __init__(self, in_channels, out_channels=0, mid_channels=0, kernel_size=3, do_bn=True, dropout_ratio=0, activation=F.relu):
        super(ResBlockBN, self).__init__()
        self.do_bn = do_bn
        self.dropout_ratio = dropout_ratio
        self.activation = __select_activation__(activation)
        if out_channels <= 0:
            out_channels = in_channels
        if mid_channels <= 0:
            mid_channels = in_channels // 4
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        if self.do_bn:
            self.bn1 = nn.BatchNorm2d(num_features=mid_channels)
            self.bn2 = nn.BatchNorm2d(num_features=mid_channels)
            self.bn3 = nn.BatchNorm2d(num_features=in_channels)
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)
        if in_channels != out_channels:
            self.use_resconv = True
            self.resconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            if self.do_bn:
                self.resbn = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.use_resconv = False

    def __call__(self, x):
        if self.do_bn:
            h = self.activation(self.bn1(self.conv1(x))) # 畳込み＆バッチ正規化＆活性化関数
            h = self.activation(self.bn2(self.conv2(h))) # 畳込み＆バッチ正規化＆活性化関数
            h = self.bn3(self.conv3(h)) # 畳込み＆バッチ正規化
        else:
            h = self.activation(self.conv1(x)) # 畳込み＆活性化関数
            h = self.activation(self.conv2(h)) # 畳込み＆活性化関数
            h = self.conv3(h) # 畳込み
        h = self.activation(h + x) # 加算＆活性化関数
        if self.use_resconv:
            if self.do_bn:
                h = self.activation(self.resbn(self.resconv(h))) # チャンネル数を変更するための畳込み＆バッチ正規化
            else:
                h = self.activation(self.resconv(h)) # チャンネル数を変更するための畳込み
        if self.dropout_ratio != 0:
            h = self.dropout(h) # ドロップアウト
        return h


# 主として GAN のディスクリミネータで用いる Minibatch Standard Deviation 層
class MinibatchStdev(nn.Module):

    def __init__(self):
        super(MinibatchStdev, self).__init__()

    def __call__(self, h):
        size = h.size()
        size = (size[0], 1, size[2], size[3])
        return torch.cat((h, torch.mean(torch.std(h, dim=0, unbiased=False)).repeat(size)), dim=1)


# 主として GAN のディスクリミネータで用いる Minibatch Discrimination 層
# 引用元: https://gist.github.com/t-ae/732f78671643de97bbe2c46519972491
class MinibatchDiscrimination(nn.Module):

    def __init__(self, in_features, out_features, kernel_dims=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        nn.init.normal_(self.T, 0, 1)

    def __call__(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)
        M = matrices.unsqueeze(0) # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3) # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3) # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1) # NxB, subtract self distance
        return torch.cat([x, o_b], 1)


# Conv のバッチ正規化を Spectral Normalization に置き換えたもの
class ConvSN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=-1, dropout_ratio=0, activation=F.relu):
        super(ConvSN, self).__init__()
        if padding < 0:
            padding = (kernel_size - 1) // 2
        self.dropout_ratio = dropout_ratio
        self.activation = __select_activation__(activation)
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)

    def __call__(self, x):
        h = self.conv(x) # 畳込み
        if not self.activation is None:
            h = self.activation(h) # 活性化関数
        if self.dropout_ratio != 0:
            h = self.dropout(h) # ドロップアウト
        return h


# ConvHalf のバッチ正規化を Spectral Normalization に置き換えたもの
class ConvHalfSN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, dropout_ratio=0, activation=F.relu):
        super(ConvHalfSN, self).__init__()
        self.conv = ConvSN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=(kernel_size-1)//2,
            dropout_ratio=dropout_ratio,
            activation=activation)

    def __call__(self, x):
        return self.conv(x)


# Deconv のバッチ正規化を Spectral Normalization に置き換えたもの
class DeconvSN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=-1, dropout_ratio=0, activation=F.relu):
        super(DeconvSN, self).__init__()
        if padding < 0:
            padding = (kernel_size - 1) // 2
        self.dropout_ratio = dropout_ratio
        self.activation = __select_activation__(activation)
        self.deconv = nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)

    def __call__(self, x):
        h = self.deconv(x) # 逆畳込み
        if not self.activation is None:
            h = self.activation(h) # 活性化関数
        if self.dropout_ratio != 0:
            h = self.dropout(h) # ドロップアウト
        return h


# FC のバッチ正規化を Spectral Normalization に置き換えたもの
class FCSN(nn.Module):

    def __init__(self, in_features, out_features, dropout_ratio=0, activation=F.relu):
        super(FCSN, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.activation = __select_activation__(activation)
        self.fc = nn.utils.spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))
        if dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)

    def __call__(self, x):
        h = self.fc(x) # 全結合
        if not self.activation is None:
            h = self.activation(h) # 活性化関数
        if self.dropout_ratio != 0:
            h = self.dropout(h) # ドロップアウト
        return h


# PixShuffleのバッチ正規化を Spectral Normalization に置き換えたもの
class PixShuffleSN(nn.Module):

    def __init__(self, in_channels, out_channels, scale=2, kernel_size=3, dropout_ratio=0, activation=F.relu):
        super(PixShuffleSN, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.activation = __select_activation__(activation)
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels*scale*scale, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2))
        self.ps = nn.PixelShuffle(scale)
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)

    def __call__(self, x):
        h = self.conv(x) # 畳込み
        if not self.activation is None:
            h = self.activation(h) # 活性化関数
        h = self.ps(h) # pixel shuffle
        if self.dropout_ratio != 0:
            h = self.dropout(h) # ドロップアウト
        return h
