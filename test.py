import torch
from transformers import T5Tokenizer
from torchvision import transforms
from data import CocoCaptions

tokenizer = T5Tokenizer.from_pretrained('t5-small')

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([224, 224]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
])

val_data = CocoCaptions(root='data/coco/images/val2014', 
        annFile='data/coco/annotations/captions_val2014.json',
        transform=image_transform, target_transform=tokenizer)

pretrained_path = ''
state = torch.load(pretrained_path)
model = state['model']

x, y, img_id = val_data[0]
out = model(x, y)