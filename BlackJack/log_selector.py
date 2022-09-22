import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='Black Jack Log Selector Sample')
parser.add_argument('--in_file', type=str, default='play_log.csv', help='input filename (raw play log)')
parser.add_argument('--out_file', type=str, default='selected_log.csv', help='output filename')
args = parser.parse_args()

# ログファイルを読み込む
df = pd.read_csv(args.in_file)

# あくまで例として，
# 「プレイヤーステータス（result）が'lose', 'bust', 'surrendered'の何れでもでない」または「プレイヤーステータス（result）が'surrendered'であり，かつ行動前スコア（score）が15以上17以下」
# を満たすものだけを抽出する場合．以下のように記述する
selected_log = df[ ~(df['result'].isin(['lose', 'bust', 'surrendered'])) | ((df['result'] == 'surrendered') & (15 <= df['score']) & (df['score'] <= 17)) ]

# 抽出結果をファイルに保存
selected_log.to_csv(args.out_file, index=False)
