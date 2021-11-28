import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

# validation dataset instance / caption annotations
COCOAPI_DIR = '../cocoapi'
if False:
    coco = COCO(os.path.join(COCOAPI_DIR, 'annotations/instances_val2014.json'))
    coco_capt = COCO(os.path.join(COCOAPI_DIR, 'annotations/captions_val2014.json'))
    img_ids = list(coco.anns.keys())

    # get a random image from ids 0-9 and acquire its info
    idx = np.random.choice(img_ids)
    print(f'idx: {idx}')
    img = coco.loadImgs(coco.anns[idx]['image_id'])[0]
    img_url = img['coco_url']
    img_vis = io.imread(img_url)
    plt.imshow(img_vis)
    plt.savefig(f'plots/{idx}.jpg')

    ann = coco_capt.loadAnns(coco_capt.getAnnIds(imgIds=img['id']))
    coco_capt.showAnns(ann)
