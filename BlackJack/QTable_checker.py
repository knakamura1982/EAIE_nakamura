import argparse
from classes import Action, QTable, get_action_name


ORDER = {Action.UNDEFINED:0, Action.HIT:1, Action.STAND:2, Action.DOUBLE_DOWN:3, Action.SURRENDER:4, Action.RETRY:5}

parser = argparse.ArgumentParser(description='Black Jack Q Table Checker')
parser.add_argument('--file', type=str, default='', help='filename of Q table to be checked')
args = parser.parse_args()

if args.file != '':
    q_table = QTable(action_class=Action)
    q_table.load(args.file)
    print('State,Action,Q value')
    for k, v in sorted(q_table.table.items(), key=lambda x:(x[0][0], ORDER[x[0][1]])):
        state, action = k
        print('{},{},{}'.format(state, get_action_name(action), v))
