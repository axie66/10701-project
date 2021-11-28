import os
import sys
import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

annotation_file = 'data/coco/annotations/captions_val2014.json'
# run_dir = 'checkpoint/2021-11-26_20:40:54.244798'
run_dir = sys.argv[-1]
files = [file for file in os.listdir(run_dir) if file.startswith('pred')]
#files = [file for file in files if int(file[4:-5]) > 10]
files = ['pred20.json', 'pred19.json']
full_results = dict()

for pred_file in files:
    print('Evaluating', pred_file)
    pred_file = os.path.join(run_dir, pred_file)

    coco = COCO(annotation_file)
    with open(pred_file) as f:
        pred = json.load(f)

    if isinstance(pred, dict):
        pred = [{'image_id': int(img_id), 'caption': caption[:caption.find('</s>')].strip() if caption.endswith('</s>') else caption}
                for img_id, caption in pred.items()]

        file = open(pred_file, 'w')
        file.close()

        with open(pred_file, 'w') as f:
            json.dump(pred, f)

    coco_res = coco.loadRes(pred_file)
    coco_eval = COCOEvalCap(coco, coco_res)

    coco_eval.evaluate()
    full_results[pred_file] = dict()
    print('=' * 80)
    print(f'Results for {pred_file}')
    for metric, score in coco_eval.eval.items():
        full_results[pred_file][metric] = score
        print(f'{metric}: {score:.3f}')
    print('=' * 80)
    
res_file = os.path.join(run_dir, 'res.json')
with open(res_file, 'w+') as f:
    json.dump(full_results, f)
