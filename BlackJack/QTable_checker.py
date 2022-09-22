import argparse
from classes import QTable, get_action_name


parser = argparse.ArgumentParser(description='Black Jack Q Table Checker')
parser.add_argument('--file', type=str, default='', help='filename of Q table to be checked')
args = parser.parse_args()

if args.file != '':
    q_table = QTable()
    q_table.load(args.file)
    print('State,Action,Q value')
    for k, v in q_table.table.items():
        state, action = k
        print('{},{},{}'.format(state, get_action_name(action), v))
