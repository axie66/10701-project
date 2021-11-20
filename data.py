import torchvision.datasets as dset
from torchvision import transforms

def load_data(root, annFile):
    return dset.CocoCaptions(root=root, annFile=annFile)

if __name__ == '__main__':
    data = load_data('data', 'data/ann.json')