import argparse
from classes import QTable


parser = argparse.ArgumentParser(description='Q Table Checker')
parser.add_argument('--name', type=str, required=True, help='game name (myGame, CartPole_v1, Acrobot_v1, etc.')
parser.add_argument('--file', type=str, default='', help='filename of Q table to be checked')
args = parser.parse_args()

if args.file != '':
    if args.name == 'myGame':
        from myGame import GameField
        Action = GameField.Action
        get_action_name = GameField.get_action_name
        ORDER = {Action.UNDEFINED:0, Action.PROCEED:1, Action.DOWN:2, Action.UP:3, Action.STAY:4}
    elif args.name == 'Acrobot_v1':
        from Acrobot_v1 import Action, get_action_name
        ORDER = {Action.UNDEFINED:0, Action.APPLY_N1_TORQUE:1, Action.APPLY_0_TORQUE:2, Action.APPLY_P1_TORQUE:3}
    elif args.name == 'CartPole_v1':
        from CartPole_v1 import Action, get_action_name
        ORDER = {Action.UNDEFINED:0, Action.GO_LEFT:1, Action.GO_RIGHT:2}
    elif args.name == 'MountainCar_v0':
        from MountainCar_v0 import Action, get_action_name
        ORDER = {Action.UNDEFINED:0, Action.ACCELERATE_TO_LEFT:1, Action.DO_NOT_ACCELERATE:2, Action.ACCELERATE_TO_RIGHT:3}
    elif args.name == 'LunarLander_v2':
        from LunarLander_v2 import Action, get_action_name
        ORDER = {Action.UNDEFINED:0, Action.DO_NOTHING:1, Action.FIRE_LEFT_ENGINE:2, Action.FIRE_MAIN_ENGINE:3, Action.FIRE_RIGHT_ENGINE:4}
    else:
        exit()
    q_table = QTable(action_class=Action)
    q_table.load(args.file)
    print('State,Action,Q value')
    for k, v in sorted(q_table.table.items(), key=lambda x:(x[0][0], ORDER[x[0][1]])):
        state, action = k
        print('{},{},{}'.format(state, get_action_name(action), v))
