import torch
import numpy as np
import os
import nltk
import json
import sys
sys.path.append('vocab/')
from torch.utils.data import Dataset
from torchvision import transforms
from utils.cocoapi import COCOAPI_DIR
from pycocotools.coco import COCO
from vocab.vocabulary import Vocabulary
from collections import Counter
from PIL import Image
from tqdm import tqdm

class CocoDataset(Dataset):
    def __init__(self, batch_size=1, mode='train', word_count_min=5, 
                 vocab_path='vocab/vocab.pkl'):
        super().__init__()
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        print(f'generating {mode} dataset')
        self.word_count_min = word_count_min
        self.batch_size = batch_size
        self.vocab_path = vocab_path
        self.img_folder = os.path.join(COCOAPI_DIR, f'images/{mode}2014')
        self.ann_file = None

        if mode == 'train' or mode == 'val':
            self.ann_file = os.path.join(COCOAPI_DIR, 
                                         f'annotations/captions_{mode}2014.json')
            self.coco = COCO(self.ann_file)
            self.img_ids = list(self.coco.anns.keys())
            print(f'dataset size: {len(self.img_ids)}')
            # tokenize annotations
            self.tokens = [nltk.tokenize.word_tokenize(
                str(self.coco.anns[self.img_ids[index]]["caption"]).lower()
            ) for index in tqdm(np.arange(len(self.img_ids)))]
            self.ann_lens = [len(item) for item in self.tokens]
        elif mode == 'test':
            self.ann_file = os.path.join(COCOAPI_DIR,
                                         'annotations/image_info_test2014.json')
            self.test_info = json.loads(open(self.ann_file).read())
            self.test_paths = [item["file_name"] for item in self.test_info["images"]]
            print(f'dataset size: {len(self.test_paths)}')

        self.vocab = Vocabulary(self.word_count_min, vocab_file=self.vocab_path,
                                annotations_file=self.ann_file)

    def __len__(self):
        if self.mode == 'train' or self.mode == 'val':
            return len(self.img_ids)
        elif self.mode == 'test':
            return len(self.test_paths)

    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode == 'val':
            idx = self.img_ids[idx]
            cap = self.coco.anns[idx]["caption"]
            tokens = nltk.tokenize.word_tokenize(str(cap).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()
            img_id = self.coco.anns[idx]["image_id"]
            url = self.coco.loadImgs(img_id)[0]["file_name"]
            image = Image.open(os.path.join(self.img_folder, url)).convert("RGB")
            image = self.transform()(image)
            return image, caption

        elif self.mode == 'test':
            url = self.test_paths[idx]
            image = Image.open(os.path.join(self.img_folder, url)).convert("RGB")
            ori_image = np.array(image)
            tra_image = self.transform()(image)
            return ori_image, tra_image

    def transform(self):
        # transform training images
        return transforms.Compose([ 
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])

    def get_indices(self):
        leng = np.random.choice(self.ann_lens)
        indices = np.where([self.ann_lens[i] == leng for i in 
                            np.arange(len(self.ann_lens))])[0]
        indices = list(np.random.choice(indices, size=self.batch_size))
        return indices
