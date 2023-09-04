import sys
import torch


def print_args(args):

    # コマンドライン引数データを辞書型（dict型）に変換
    args = vars(args)

    # 引数の内容を出力
    for key, value in args.items():
        if key == 'gpu':
            dev_str = 'cuda:{0}'.format(value) if torch.cuda.is_available() and value >= 0 else 'cpu'
            print('device: {0}'.format(dev_str), file=sys.stderr)
        else:
            print('{0}: {1}'.format(key, value), file=sys.stderr)
    print('')

    if 'gpu' in args.keys():
        args['device'] = dev_str

    return args


def randn_truncated(shape, a:float=-1, b:float=1):
    u = torch.rand(shape)
    a = torch.tensor([a])
    b = torch.tensor([b])
    t2 = torch.tensor([2])
    Fa = 0.5 * (1 + torch.erf(a / torch.sqrt(t2)))
    Fb = 0.5 * (1 + torch.erf(b / torch.sqrt(t2)))
    return torch.sqrt(t2) * torch.erfinv(2 * ((Fb - Fa) * u + Fa) - 1)
