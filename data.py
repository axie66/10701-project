import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import os
import os.path
import random

class CocoCaptions(data.Dataset):
    """
    Note: Taken from torchvision datasets with slight modification.
    
    `MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    Example:

        .. code:: python

            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root = 'dir where images are',
                                    annFile = 'json annotation file',
                                    transform=transforms.ToTensor())

            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample

            print("Image Size: ", img.size())
            print(target)

        Output: ::

            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']

    """
    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        target = [ann['caption'] for ann in anns]

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, img_id

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def collate_fn(batch, sos_token_id=None, eos_token_id=None, pad_token_id=0):
        X, Y, Y_attn, img_ids = [], [], [], []
        for x, y, img_id in batch:
            X.append(x)
            # each time, randomly select a target caption
            idx = random.randint(0, len(y.input_ids)-1)
            if sos_token_id is not None and eos_token_id is not None:
                Y.append(torch.tensor([sos_token_id] + y.input_ids[idx] + [eos_token_id]))
                Y_attn.append(torch.tensor([1, 1] + y.attention_mask[idx]))
            else:
                Y.append(torch.tensor(y.input_ids[idx]))
                Y_attn.append(torch.tensor(y.attention_mask[idx]))
            img_ids.append(img_id)

        X = torch.stack(X, dim=0)
        Y = pad_sequence(Y, batch_first=True, padding_value=pad_token_id)
        Y_attn = pad_sequence(Y_attn, batch_first=True)
        img_ids = torch.tensor(img_ids)
        return X, Y, Y_attn, img_ids