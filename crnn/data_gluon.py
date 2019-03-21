import os
import numpy as np
from mxnet.gluon.data.vision import ImageFolderDataset
import warnings
from mxnet import image,nd
from .config import config
class ImageDataset(ImageFolderDataset):
    def __init__(self, root,train=True,flag = 1,transform=None):
        # split = 'train.txt' if train else 'val.txt'
        # root = path.join(root, split)
        self._train = train
        super(ImageDataset, self).__init__(root=root, flag=flag, transform=transform)
        self._root = os.path.expanduser(root)
        self._flag = flag
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self._list_images(self._root)
        self._num = self.__len__()

    def _list_images(self, root):
        self.items = []
        split = 'train.txt'if self._train else 'test.txt'
        with open(os.path.join(root,split),'r') as file:
            lines = file.readlines()
            for line in lines:
                annot = line.strip().split(' ')
                img_path = os.path.join(root,'images',annot[0])
                ext = os.path.splitext(img_path)[1]
                if ext.lower() not in self._exts:
                    warnings.warn('Ignoring %s of type %s. Only support %s'%(
                        img_path, ext, ', '.join(self._exts)))
                    continue
                # if img_path == '/home/xddz/lironghua/datasets/Syntheic_Chinese/images/72690593_1335283478.jpg':
                #     continue
                label = annot[1:]
                self.items.append((img_path, label))

    def __getitem__(self, idx):
        # print(self.items[idx][0])
        img = image.imread(self.items[idx][0], self._flag)
        img = image.imresize(img,config.img_width,config.img_height)
        img = nd.array(img, dtype='float32')
        img = nd.transpose(data=img, axes=[2,0,1])
        img -= 127.5
        img *= 0.0078125
        label = self.items[idx][1]
        label = nd.array(label,dtype='float32')
        if self._transform is not None:
            return self._transform(img, label)
        return img,label

if __name__ == '__main__':
    dataset = ImageDataset(root='/home/xddz/lironghua/datasets/Train_data_digit',train=False)
    for img,label in dataset:
        print(img.shape)
        # /home/xddz/ lironghua / datasets / Syntheic_Chinese / images / 72690593_1335283478.jpg


