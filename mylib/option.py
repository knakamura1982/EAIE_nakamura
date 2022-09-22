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
