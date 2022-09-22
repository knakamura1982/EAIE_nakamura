import argparse
from qtable import QTable


parser = argparse.ArgumentParser(description='Black Jack Q Table Checker')
parser.add_argument('--file', type=str, default='', help='filename of Q table to be checked')
args = parser.parse_args()

if args.file != '':
    q_table = QTable(n_actions=1) # n_actionsの値はダミー
    q_table.load(args.file)
    print('State,Action ID,Q value')
    for k, v in q_table.table.items():
        state, action = k
        print('{},{},{}'.format(state, action, v))
