import json
import sys

#filename = 'checkpoint/2021-11-26_16:38:23.019753/res.json'
filename = sys.argv[-1]

with open(filename, 'r') as f:
    x = json.load(f)

keys = keys = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']
for k in sorted(x, key=lambda a: int(a.split('/')[-1][4:-5])):
    print('\t'.join([k.split('/')[-1][4:-5]] + [str(x[k][key]) for key in keys]))
