import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor


# PixShuffle層の重みの初期化に使用するモジュール
# 外部から使用することは想定していない
def __ICNR__(tensor, initializer, upscale_factor=2, *args, **kwargs):
    upscale_factor_squared = upscale_factor * upscale_factor
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared, *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)


# 活性化関数を選択するモジュール
# 外部から使用することは想定していない
def __select_activation__(activation):
    activation = activation.lower()
    if activation == 'e' or activation == 'elu':
        act = F.elu
    elif activation == 'g' or activation == 'gelu':
        act = F.gelu
    elif activation == 'l' or activation == 'leaky-relu' or activation == 'leaky_relu':
        act = F.leaky_relu
    elif activation == 'p' or activation == 'prelu':
        act = F.prelu
    elif activation == 'r' or activation == 'relu':
        act = F.relu
    elif activation == 's' or activation == 'sigmoid':
        act = torch.sigmoid
    elif activation == 't' or activation == 'tanh':
        act = torch.tanh
    else:
        act = None
    return act


# ベース層 layer に正規化層と活性化関数を付加するモジュール
# 外部から使用することは想定していない
def __wrap_layer__(layer:nn.Module, normalization:str='none', activation:str='none', pre_act:bool=False, **kwargs):

    # 活性化関数の作成
    activation = activation.lower()
    if activation == 'e' or activation == 'elu':
        act = nn.ELU()
    elif activation == 'g' or activation == 'gelu':
        act = nn.GELU()
    elif activation == 'l' or activation == 'leaky-relu' or activation == 'leaky_relu':
        act = nn.LeakyReLU()
    elif activation == 'p' or activation == 'prelu':
        act = nn.PReLU()
    elif activation == 'r' or activation == 'relu':
        act = nn.ReLU()
    elif activation == 's' or activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 't' or activation == 'tanh':
        act = nn.Tanh()
    else:
        act = None

    # 正規化層の作成
    normalization = normalization.lower()
    if normalization == 'b' or normalization == 'batch':
        if type(layer) == torch.nn.modules.conv.Conv3d or type(layer) == torch.nn.modules.conv.ConvTranspose3d:
            norm = nn.BatchNorm3d(num_features=kwargs['num_features'])
        elif type(layer) == torch.nn.modules.conv.Conv2d or type(layer) == torch.nn.modules.conv.ConvTranspose2d:
            norm = nn.BatchNorm2d(num_features=kwargs['num_features'])
        else:
            norm = nn.BatchNorm1d(num_features=kwargs['num_features'])
    elif normalization == 'g' or normalization == 'group':
        norm = nn.GroupNorm(num_channels=kwargs['num_features'], num_groups=kwargs['num_groups'])
    elif normalization == 'i' or normalization == 'instance':
        if type(layer) == torch.nn.modules.conv.Conv3d or type(layer) == torch.nn.modules.conv.ConvTranspose3d:
            norm = nn.InstanceNorm3d(num_features=kwargs['num_features'])
        elif type(layer) == torch.nn.modules.conv.Conv2d or type(layer) == torch.nn.modules.conv.ConvTranspose2d:
            norm = nn.InstanceNorm2d(num_features=kwargs['num_features'])
        elif type(layer) == torch.nn.modules.conv.Conv1d or type(layer) == torch.nn.modules.conv.ConvTranspose1d:
            norm = nn.InstanceNorm1d(num_features=kwargs['num_features'])
        else:
            norm = None
    elif normalization == 'l' or normalization == 'layer':
        norm = nn.LayerNorm(normalized_shape=kwargs['normalized_shape'])
    elif normalization == 's' or normalization == 'spectral':
        layer = nn.utils.spectral_norm(layer)
        norm = None
    else:
        norm = None

    # 正規化層と活性化関数の付加
    if act is None and norm is None:
        return layer
    modules = []
    if pre_act:
        # pre-activation の場合
        if norm is not None:
            modules.append(norm)
        if act is not None:
            modules.append(act)
        modules.append(layer)
    else:
        # post-activation の場合
        modules.append(layer)
        if norm is not None:
            modules.append(norm)
        if act is not None:
            modules.append(act)
    return nn.Sequential(*modules)


# 畳込み + 正規化 + 活性化関数を行う層
# kernel_size を奇数に設定した上で stride と padding を指定しなければ（デフォルト値にすれば）, 出力マップと入力マップは同じサイズになる
#   - in_channels: 入力マップのチャンネル数
#   - out_channels: 出力マップのチャンネル数
#   - kernel_size: カーネルサイズ（デフォルトで 3 になる）
#   - stride: ストライド幅（デフォルトで 1 になる）
#   - padding: パディングサイズ（デフォルトで (kernel_size-1)/2 になる）
#   - dropout_ratio: 0 以外ならその割合でドロップアウト処理を実行（デフォルトでは 0, すなわちドロップアウトなし）
#   - normalization: 正規化手法（'none', 'batch', 'spectral', 'layer' など. デフォルトでは 'batch' ）
#   - activation: 活性化関数（デフォルトでは ReLU になる）
class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=-1, dropout_ratio=0, normalization='batch', activation='relu', **kwargs):
        super(Conv, self).__init__()
        if type(padding) == int and padding < 0:
            padding = (kernel_size - 1) // 2
        self.dropout_ratio = dropout_ratio
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv = __wrap_layer__(self.conv, normalization=normalization, activation=activation, num_features=out_channels, **kwargs) # 正規化層と活性化関数の付加
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)

    def __call__(self, x):
        h = self.conv(x) # 畳込み + 正規化 + 活性化関数
        if self.dropout_ratio != 0:
            h = self.dropout(h) # ドロップアウト
        return h


# 畳込み + 正規化 + 活性化関数を行う層
# 上記 Conv のラッパークラスで, 出力マップのサイズが入力マップの半分になるように stride と padding が自動設定される
# 入力マップのサイズは縦横ともに偶数であり, kernel_size にも偶数が指定されることを前提としている（そうでない場合の動作は保証外）
# なお, kernel_size のデフォルト値は 4
class ConvHalf(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, dropout_ratio=0, normalization='batch', activation='relu', **kwargs):
        super(ConvHalf, self).__init__()
        self.conv = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=(kernel_size-1)//2,
            dropout_ratio=dropout_ratio,
            normalization=normalization,
            activation=activation,
            **kwargs)

    def __call__(self, x):
        return self.conv(x)


# 逆畳込み + 正規化 + 活性化関数を行う層
# kernel_size を偶数に設定した上で stride と padding を指定しなければ（デフォルト値にすれば）, 出力マップのサイズは入力マップのちょうど2倍になる
#   - in_channels: 入力マップのチャンネル数
#   - out_channels: 出力マップのチャンネル数
#   - kernel_size: カーネルサイズ（デフォルトで 4 になる）
#   - stride: ストライド幅（デフォルトで 2 になる）
#   - padding: パディングサイズ（デフォルトで (kernel_size-1)/2 になる）
#   - dropout_ratio: 0 以外ならその割合でドロップアウト処理を実行（デフォルトでは 0, すなわちドロップアウトなし）
#   - normalization: 正規化手法（'none', 'batch', 'spectral', 'layer' など. デフォルトでは 'batch' ）
#   - activation: 活性化関数（デフォルトでは ReLU になる）
class Deconv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=-1, dropout_ratio=0, normalization='batch', activation='relu', **kwargs):
        super(Deconv, self).__init__()
        if type(padding) == int and padding < 0:
            padding = (kernel_size - 1) // 2
        self.dropout_ratio = dropout_ratio
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.deconv = __wrap_layer__(self.deconv, normalization=normalization, activation=activation, num_features=out_channels, **kwargs) # 正規化層と活性化関数の付加
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)

    def __call__(self, x):
        h = self.deconv(x) # 逆畳込み + 正規化 + 活性化関数
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


# 畳込み + 正規化 + 活性化関数 + Pixel Shuffle を行う層
# 畳込みにおける stride と padding は, 出力マップのサイズが入力マップのサイズの scale 倍になるように自動設定する
#   - in_channels: 入力マップのチャンネル数
#   - out_channels: 出力マップのチャンネル数
#   - scale: Pixel Shuffle におけるスケールファクタ（デフォルトで 2 になる）
#   - kernel_size: カーネルサイズ（奇数のみ可, 偶数の場合の動作は保証外. デフォルトで 3 になる）
#   - dropout_ratio: 0 以外ならその割合でドロップアウト処理を実行（デフォルトでは 0, すなわちドロップアウトなし）
#   - normalization: 正規化手法（'none', 'batch', 'spectral', 'layer' など. デフォルトでは 'batch' ）
#   - activation: 活性化関数（デフォルトでは ReLU になる）
class PixShuffle(nn.Module):

    def __init__(self, in_channels, out_channels, scale=2, kernel_size=3, dropout_ratio=0, normalization='batch', activation='relu', **kwargs):
        super(PixShuffle, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels*(scale**2), kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.conv.weight.data.copy_(__ICNR__(self.conv.weight, initializer=nn.init.kaiming_normal_, upscale_factor=scale))
        self.conv = __wrap_layer__(self.conv, normalization=normalization, activation=activation, num_features=out_channels*(scale**2), **kwargs) # 正規化層と活性化関数の付加
        self.ps = nn.PixelShuffle(scale)
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)

    def __call__(self, x):
        h = self.conv(x) # 畳込み + 正規化 + 活性化関数
        h = self.ps(h) # pixel shuffle
        if self.dropout_ratio != 0:
            h = self.dropout(h) # ドロップアウト
        return h


# 全結合 + 正規化 + 活性化関数を行う層
#   - in_features: 入力のパーセプトロン数
#   - out_features: 出力のパーセプトロン数
#   - dropout_ratio: 0 以外ならその割合でドロップアウト処理を実行（デフォルトでは 0, すなわちドロップアウトなし）
#   - normalization: 正規化手法（'none', 'batch', 'spectral', 'layer' など. デフォルトでは 'batch' ）
#   - activation: 活性化関数（デフォルトでは ReLU になる）
class FC(nn.Module):

    def __init__(self, in_features, out_features, dropout_ratio=0, normalization='batch', activation='relu', **kwargs):
        super(FC, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.fc = __wrap_layer__(self.fc, normalization=normalization, activation=activation, num_features=out_features, **kwargs) # 正規化層と活性化関数の付加
        if dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)

    def __call__(self, x):
        h = self.fc(x) # 全結合 + 正規化 + 活性化関数
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


# Conditional Batch Normalization
class ConditionalBatchNorm2d(nn.Module):

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features)
        self.gamma_embed = nn.Linear(num_classes, num_features, bias=False)
        self.beta_embed = nn.Linear(num_classes, num_features, bias=False)

    def __call__(self, x, y):
        out = self.bn(x)
        gamma = self.gamma_embed(y) + 1
        beta = self.beta_embed(y)
        return gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)


# Adaptive Instance Normalization
#   - x: コンテンツ特徴マップ
#   - y: スタイル特徴マップ
class AdaIN(nn.Module):

    def __init__(self, in_channels_content, in_channels_style):
        super(AdaIN, self).__init__()
        if in_channels_content == in_channels_style:
            self.conv = None
        else:
            self.conv = Conv(in_channels=in_channels_style, out_channels=in_channels_content, kernel_size=1, stride=1, padding=0, normalization='none', activation='none')

    def __call__(self, x, y):
        if self.conv is not None:
            y = self.conv(y)
        xu = torch.mean(x, dim=(2, 3), keepdim=True) # コンテンツ特徴マップのチャンネルごとの平均
        yu = torch.mean(y, dim=(2, 3), keepdim=True) # スタイル特徴マップのチャンネルごとの平均
        xs = torch.std(x, dim=(2, 3), unbiased=False, keepdim=True) # コンテンツ特徴マップのチャンネルごとの標準偏差
        ys = torch.std(y, dim=(2, 3), unbiased=False, keepdim=True) # スタイル特徴マップのチャンネルごとの標準偏差
        h = ys * ((x - xu) / xs) + yu
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
#   - model: バックボーンとして使用する有名モデル（例えば models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) など）
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

    def train(self, mode: bool = True):
        if self.finetune:
            super().train(mode)
            self.backbone.train(mode)
        else:
            super().train(False)
            self.backbone.train(False)
        return self

    def eval(self):
        return self.train(False)

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
#   - dropout_ratio: 0 以外ならその割合でドロップアウト処理を実行（デフォルトでは 0, すなわちドロップアウトなし）
#   - normalization: 正規化手法（'none', 'batch', 'spectral', 'layer' など. デフォルトでは 'batch' ）
#   - activation: 活性化関数（デフォルトでは ReLU になる）
class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels=0, kernel_size=3, dropout_ratio=0, normalization='batch', activation='relu', **kwargs):
        super(ResBlock, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.activation = __select_activation__(activation)
        if out_channels <= 0:
            out_channels = in_channels
        self.conv1 = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2,
                          normalization=normalization, activation=activation, **kwargs)
        self.conv2 = Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2,
                          normalization=normalization, activation='none', **kwargs)
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)
        if in_channels == out_channels:
            self.shortcut = None
        else:
            normalization = normalization.lower()
            if normalization == 's' or normalization == 'spectral':
                self.shortcut = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, normalization='spectral', activation='none')
            else:
                self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def __call__(self, x):
        h = self.conv1(x) # 畳込み + 正規化 + 活性化関数
        h = self.conv2(h) # 畳込み + 正規化
        if self.shortcut is not None:
            x = self.shortcut(x)
        h = h + x # ショートカットを加算
        if self.activation is not None:
            h = self.activation(h) # 活性化関数
        if self.dropout_ratio != 0:
            h = self.dropout(h) # ドロップアウト
        return h


# Pre-activation 型 Residual Block 層
# 畳込みにおける stride と padding は, 出力マップが入力マップと同じサイズになるように自動決定する
#   - in_channels: 入力マップのチャンネル数
#   - out_channels: 出力マップのチャンネル数（0 以下の時は自動的に in_channels と同じ値を設定，デフォルトでは 0 ）
#   - kernel_size: カーネルサイズ（奇数のみ可, 偶数の場合の動作は保証外. デフォルトで 3 になる）
#   - dropout_ratio: 0 以外ならその割合でドロップアウト処理を実行（デフォルトでは 0, すなわちドロップアウトなし）
#   - normalization: 正規化手法（'none', 'batch', 'spectral', 'layer' など. デフォルトでは 'batch' ）
#   - activation: 活性化関数（デフォルトでは ReLU になる）
class ResBlockPA(nn.Module):

    def __init__(self, in_channels, out_channels=0, kernel_size=3, dropout_ratio=0, normalization='batch', activation='relu', **kwargs):
        super(ResBlockPA, self).__init__()
        self.dropout_ratio = dropout_ratio
        if out_channels <= 0:
            out_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.conv1 = __wrap_layer__(self.conv1, normalization=normalization, activation=activation, pre_act=True, num_features=in_channels, **kwargs) # 正規化層と活性化関数の付加
        self.conv2 = __wrap_layer__(self.conv2, normalization=normalization, activation=activation, pre_act=True, num_features=out_channels, **kwargs) # 正規化層と活性化関数の付加
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)
        if in_channels == out_channels:
            self.shortcut = None
        else:
            normalization = normalization.lower()
            if normalization == 's' or normalization == 'spectral':
                self.shortcut = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, normalization='spectral', activation='none')
            else:
                self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def __call__(self, x):
        h = self.conv1(x) # 正規化 + 活性化関数 + 畳込み
        h = self.conv2(h) # 正規化 + 活性化関数 + 畳込み
        if self.shortcut is not None:
            x = self.shortcut(x)
        h = h + x # ショートカットを加算
        if self.dropout_ratio != 0:
            h = self.dropout(h) # ドロップアウト
        return h


# Bottle-Neck 型 Residual Block 層
# 畳込みにおける stride と padding は, 出力マップが入力マップと同じサイズになるように自動決定する
#   - in_channels: 入力マップのチャンネル数
#   - out_channels: 出力マップのチャンネル数（0 以下の時は自動的に in_channels と同じ値を設定，デフォルトでは 0 ）
#   - mid_channels: 中間層のマップのチャンネル数（0 以下の時は自動的に in_channels // 4 を設定, デフォルトでは 0 ）
#   - kernel_size: カーネルサイズ（奇数のみ可, 偶数の場合の動作は保証外. デフォルトで 3 になる）
#   - dropout_ratio: 0 以外ならその割合でドロップアウト処理を実行（デフォルトでは 0, すなわちドロップアウトなし）
#   - normalization: 正規化手法（'none', 'batch', 'spectral', 'layer' など. デフォルトでは 'batch' ）
#   - activation: 活性化関数（デフォルトでは ReLU になる）
class ResBlockBN(nn.Module):

    def __init__(self, in_channels, out_channels=0, mid_channels=0, kernel_size=3, dropout_ratio=0, normalization='batch', activation='relu', **kwargs):
        super(ResBlockBN, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.activation = __select_activation__(activation)
        if out_channels <= 0:
            out_channels = in_channels
        if mid_channels <= 0:
            mid_channels = in_channels // 4
        self.conv1 = Conv(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0,
                          normalization=normalization, activation=activation, **kwargs)
        self.conv2 = Conv(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2,
                          normalization=normalization, activation=activation, **kwargs)
        self.conv3 = Conv(in_channels=mid_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0,
                          normalization=normalization, activation='none', **kwargs)
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)
        if in_channels == out_channels:
            self.shortcut = None
        else:
            normalization = normalization.lower()
            if normalization == 's' or normalization == 'spectral':
                self.shortcut = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, normalization='spectral', activation='none')
            else:
                self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def __call__(self, x):
        h = self.conv1(x) # 畳込み + 正規化 + 活性化関数
        h = self.conv2(h) # 畳込み + 正規化 + 活性化関数
        h = self.conv3(h) # 畳込み + 正規化
        if self.shortcut is not None:
            x = self.shortcut(x)
        h = h + x # ショートカットを加算
        if self.activation is not None:
            h = self.activation(h) # 活性化関数
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


# GAN用の Discriminator Augmentation を実現するクラス
class DiscriminatorAugmentation(nn.Module):

    # H: 入力画像の縦幅
    # W: 入力画像の横幅
    # p_hflip: 左右反転の実行確率
    # p_vflip: 上下反転の実行確率（ H != W の場合のみ ）
    # p_rot: 回転（90, 180, 270度）の実行確率（ H == W の場合のみ ）
    def __init__(self, H, W, p_hflip, p_vflip, p_rot):
        super(DiscriminatorAugmentation, self).__init__()
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot = p_rot
        if H == W:
            self.is_square = True
        else:
            self.is_square = False

    def forward(self, *args):
        ret = args
        if torch.rand(1) < self.p_hflip:
            ret = [transforms.functional.hflip(x) for x in args]
        if self.is_square:
            if torch.rand(1) < self.p_rot:
                angle = int(90 * (torch.randint(3, (1,)) + 1))
                ret = [transforms.functional.rotate(x, angle, fill=0) for x in ret]
        else:
            if torch.rand(1) < self.p_vflip:
                ret = [transforms.functional.vflip(x) for x in ret]
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)
