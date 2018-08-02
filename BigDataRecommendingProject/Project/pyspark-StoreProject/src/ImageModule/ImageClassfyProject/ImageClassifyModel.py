from __future__ import division
import os
import time
import math
import numpy as np
import scipy.misc
import argparse
import tensorflow as tf
from glob import glob
import torch
from torch.autograd import Variable
from tqdm import tqdm
import numbers
import torchvision.transforms.functional as F
from torchvision.transforms import transforms
import json
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from augmentation import HorizontalFlip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from misc import FurnitureDataset, preprocess, preprocess_with_augmentation, NB_CLASSES, normalize_05
%load_ext autoreload
%autoreload 2
%matplotlib inline

import os
import json
from torch.utils.data import DataLoader
from torchvision import transforms

import models
import utils
from augmentation import five_crops, HorizontalFlip, make_transforms
from misc import FurnitureDataset, preprocess, NB_CLASSES, preprocess_hflip, normalize_05, normalize_torch
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
import seaborn as sns; sns.set_style("whitegrid")

from misc import FurnitureDataset, preprocess


'''
由于这次数据量太大，本人机器无法做到分类，除了总结一些常用的图像领域神经网络，我在网上仿照了一位大神做图像分类的思想。
当然他的机器要求是最低64G内存才可以跑通。我会将这段代码思想分析出来，方便以后做类似图像处理问题时可以使用。
'''

'''
设置GPU
'''
use_gpu = torch.cuda.is_available()


'''
模块1:加载数据
'''
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

'''
数据转化
'''
def ParseData(data_file):
  ann = {}
  if 'train' in data_file or 'validation' in data_file:
      _ann = json.load(open(data_file))['annotations']
      for a in _ann:
        ann[a['image_id']] = a['label_id']

  key_url_list = []
  j = json.load(open(data_file))
  images = j['images']
  for item in images:
    assert len(item['url']) == 1
    url = item['url'][0]
    id_ = item['image_id']
    if id_ in ann:
        id_ = "{}_{}".format(id_, ann[id_])
    key_url_list.append((id_, url))
  return key_url_list

'''
下载图片
'''
def DownloadImage(key_url):
  out_dir = sys.argv[2]
  (key, url) = key_url
  filename = os.path.join(out_dir, '%s.jpg' % key)

  if os.path.exists(filename):
    print('Image %s already exists. Skipping download.' % filename)
    return

  try:
    #print('Trying to get %s.' % url)
    http = urllib3.PoolManager(100)
    response = http.request('GET', url)
    image_data = response.data
  except:
    print('Warning: Could not download image %s from %s' % (key, url))
    return

  try:
    pil_image = Image.open(BytesIO(image_data))
  except:
    print('Warning: Failed to parse image %s %s' % (key,url))
    return

  try:
    pil_image_rgb = pil_image.convert('RGB')
  except:
    print('Warning: Failed to convert image %s to RGB' % key)
    return

  try:
    pil_image_rgb.save(filename, format='JPEG', quality=90)
  except:
    print('Warning: Failed to save image %s' % filename)
    return

'''
多线程下载
'''
def Run():
  if len(sys.argv) != 3:
    print('Syntax: %s <train|validation|test.json> <output_dir/>' % sys.argv[0])
    sys.exit(0)
  (data_file, out_dir) = sys.argv[1:]

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  key_url_list = ParseData(data_file)
  pool = multiprocessing.Pool(processes=100)

  with tqdm(total=len(key_url_list)) as t:
    for _ in pool.imap_unordered(DownloadImage, key_url_list):
      t.update(1)


'''
模块2:工具类
'''

'''
平均值，写这个公共类是为了我们最后的取均值权重，注意用@property
'''
class RunningMean:
    def __init__(self, value=0, count=0):
        self.total_value = value
        self.count = count

    def update(self, value, count=1):
        self.total_value += value
        self.count += count

    @property
    def value(self):
        if self.count:
            return self.total_value / self.count
        else:
            return float("inf")

    def __str__(self):
        return str(self.value)

'''
这是个进度条功能的拼接功能
'''
def predict(model, dataloader):
    all_labels = []
    all_outputs = []
    model.eval()

    pbar = tqdm(dataloader, total=len(dataloader))
    for inputs, labels in pbar:
        all_labels.append(labels)

        inputs = Variable(inputs, volatile=True)
        if use_gpu:
            inputs = inputs.cuda()

        outputs = model(inputs)
        all_outputs.append(outputs.data.cpu())

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    if use_gpu:
        all_labels = all_labels.cuda()
        all_outputs = all_outputs.cuda()

    return all_labels, all_outputs

'''
图像的维度拼接，相当于tensorflow的concat
'''
def safe_stack_2array(acc, a):
    a = a.unsqueeze(-1)
    if acc is None:
        return a
    return torch.cat((acc, a), dim=acc.dim() - 1)

'''
也是跟依次拼接功能
'''
def predict_tta(model, dataloaders):
    prediction = None
    lx = None
    for dataloader in dataloaders:
        lx, px = predict(model, dataloader)
        prediction = safe_stack_2array(prediction, px)

    return lx, prediction


'''
模块三.图像增强与特征工程，我认为图像增强可以理解为特征工程里了。
'''

'''
图像翻转
'''
class HorizontalFlip(object):
    def __call__(self, img):
        return F.hflip(img)

'''
这是图像增强类似上下左右交叉验证的思路，选取中心点是哪个，就截取哪个。我们这里先管他叫交叉验证。
也就是将上下左右四个方向图片依次进行交叉比较。
'''
def five_crop(img, size, crop_pos):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    if crop_pos == 0:
        return img.crop((0, 0, crop_w, crop_h))
    elif crop_pos == 1:
        return img.crop((w - crop_w, 0, w, crop_h))
    elif crop_pos == 2:
        return img.crop((0, h - crop_h, crop_w, h))
    elif crop_pos == 3:
        return img.crop((w - crop_w, h - crop_h, w, h))
    else:
        return F.center_crop(img, (crop_h, crop_w))

'''
属于将图像交叉验证进行参数化。
'''
class FiveCropParametrized(object):
    def __init__(self, size, crop_pos):
        self.size = size
        self.crop_pos = crop_pos
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return five_crop(img, self.size, self.crop_pos)

'''
将多个图像交叉验证
'''
def five_crops(size):
    return [FiveCropParametrized(size, i) for i in range(5)]

'''
图像转换
'''
def make_transforms(first_part, second_part, inners):
    return [transforms.Compose(first_part + [inner] + second_part) for inner in inners]


'''
读取数据
'''
class FurnitureDataset(Dataset):
    def __init__(self, preffix: str, transform=None):
        self.preffix = preffix
        if preffix == 'val':
            path = 'validation'
        else:
            path = preffix
        path = f'data/{path}.json'
        self.transform = transform
        img_idx = {int(p.name.split('.')[0])
                   for p in Path(f'tmp/{preffix}').glob('*.jpg')}
        data = json.load(open(path))
        if 'annotations' in data:
            data = pd.DataFrame(data['annotations'])
        else:
            data = pd.DataFrame(data['images'])
        self.full_data = data
        nb_total = data.shape[0]
        data = data[data.image_id.isin(img_idx)].copy()
        data['path'] = data.image_id.map(lambda i: f"tmp/{preffix}/{i}.jpg")
        self.data = data
        print(f'[+] dataset `{preffix}` loaded {data.shape[0]} images from {nb_total}')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row['path'])
        if self.transform:
            img = self.transform(img)
        target = row['label_id'] - 1 if 'label_id' in row else -1
        return img, target

'''
设置全局变量的方差和标准差
'''
normalize_torch = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
normalize_05 = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

'''
图像预处理，修改尺寸的操作。
'''
def preprocess(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])


'''
图像预处理后进行翻转操作
'''
def preprocess_hflip(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        HorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

'''
图像预处理，修改尺寸，这里是扩大图片。
'''
def preprocess_with_augmentation(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20)),
        transforms.RandomRotation(15, expand=True),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor(),
        normalize
    ])


'''
模块四.构建 inceptionv4算法，根据我们介绍的CNN网络，这是个很好的图像处理算法。具有残差结构，又有ImputModel结构可以特征拼接。我认为日后可以尝试用在数据挖掘。
'''
BATCH_SIZE = 16
IMAGE_SIZE = 299

'''
调用该神经网络
'''
def get_model_inceptionv4():
    print('[+] loading model... ', end='', flush=True)
    model = models.inceptionv4_finetune(NB_CLASSES)
    if use_gpu:
        model.cuda()
    print('done')
    return model

'''
训练该神经网络
'''
def train_inceptionv4():
    train_dataset = FurnitureDataset('train', transform=preprocess_with_augmentation(normalize_05, IMAGE_SIZE))
    val_dataset = FurnitureDataset('val', transform=preprocess(normalize_05, IMAGE_SIZE))
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=8,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)
    validation_data_loader = DataLoader(dataset=val_dataset, num_workers=1,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)

    model = get_model_inceptionv4()

    criterion = nn.CrossEntropyLoss().cuda()

    nb_learnable_params = sum(p.numel() for p in model.fresh_params())
    print(f'[+] nb learnable params {nb_learnable_params}')

    lx, px = utils.predict(model, validation_data_loader)
    min_loss = criterion(Variable(px), Variable(lx)).data[0]

    lr = 0
    patience = 0
    for epoch in range(20):
        print(f'epoch {epoch}')
        if epoch == 1:
            lr = 0.00003
            print(f'[+] set lr={lr}')
        if patience == 2:
            patience = 0
            model.load_state_dict(torch.load('inception4_052382.pth'))
            lr = lr / 10
            print(f'[+] set lr={lr}')
        if epoch == 0:
            lr = 0.001
            print(f'[+] set lr={lr}')
            optimizer = torch.optim.Adam(model.fresh_params(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

        running_loss = RunningMean()
        running_score = RunningMean()

        model.train()
        pbar = tqdm(training_data_loader, total=len(training_data_loader))
        for inputs, labels in pbar:
            batch_size = inputs.size(0)

            inputs = Variable(inputs)
            labels = Variable(labels)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, dim=1)

            loss = criterion(outputs, labels)
            running_loss.update(loss.data[0], 1)
            running_score.update(torch.sum(preds != labels.data), batch_size)

            loss.backward()
            optimizer.step()

            pbar.set_description(f'{running_loss.value:.5f} {running_score.value:.3f}')
        print(f'[+] epoch {epoch} {running_loss.value:.5f} {running_score.value:.3f}')

        lx, px = utils.predict(model, validation_data_loader)
        log_loss = criterion(Variable(px), Variable(lx))
        log_loss = log_loss.data[0]
        _, preds = torch.max(px, dim=1)
        accuracy = torch.mean((preds != lx).float())
        print(f'[+] val {log_loss:.5f} {accuracy:.3f}')

        if log_loss < min_loss:
            torch.save(model.state_dict(), 'inception4_052382.pth')
            print(f'[+] val score improved from {min_loss:.5f} to {log_loss:.5f}. Saved!')
            min_loss = log_loss
            patience = 0
        else:
            patience += 1


'''
模块五.构建 densenet16网络结构
'''

BATCH_SIZE = 16
IMAGE_SIZE = 224


def get_model():
    print('[+] loading model... ', end='', flush=True)
    model = models.densenet161_finetune(NB_CLASSES)
    if use_gpu:
        model.cuda()
    print('done')
    return model


def train():
    train_dataset = FurnitureDataset('train', transform=preprocess_with_augmentation(normalize_torch, IMAGE_SIZE))
    val_dataset = FurnitureDataset('val', transform=preprocess(normalize_torch, IMAGE_SIZE))
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=8,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)
    validation_data_loader = DataLoader(dataset=val_dataset, num_workers=1,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)

    model = get_model()

    criterion = nn.CrossEntropyLoss().cuda()

    nb_learnable_params = sum(p.numel() for p in model.fresh_params())
    print(f'[+] nb learnable params {nb_learnable_params}')

    lx, px = utils.predict(model, validation_data_loader)
    min_loss = criterion(Variable(px), Variable(lx)).data[0]

    lr = 0
    patience = 0
    for epoch in range(20):
        print(f'epoch {epoch}')
        if epoch == 1:
            lr = 0.00003
            print(f'[+] set lr={lr}')
        if patience == 2:
            patience = 0
            model.load_state_dict(torch.load('densenet161_15130.pth'))
            lr = lr / 10
            print(f'[+] set lr={lr}')
        if epoch == 0:
            lr = 0.001
            print(f'[+] set lr={lr}')
            optimizer = torch.optim.Adam(model.fresh_params(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

        running_loss = RunningMean()
        running_score = RunningMean()

        model.train()
        pbar = tqdm(training_data_loader, total=len(training_data_loader))
        for inputs, labels in pbar:
            batch_size = inputs.size(0)

            inputs = Variable(inputs)
            labels = Variable(labels)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, dim=1)

            loss = criterion(outputs, labels)
            running_loss.update(loss.data[0], 1)
            running_score.update(torch.sum(preds != labels.data), batch_size)

            loss.backward()
            optimizer.step()

            pbar.set_description(f'{running_loss.value:.5f} {running_score.value:.3f}')
        print(f'[+] epoch {epoch} {running_loss.value:.5f} {running_score.value:.3f}')

        lx, px = utils.predict(model, validation_data_loader)
        log_loss = criterion(Variable(px), Variable(lx))
        log_loss = log_loss.data[0]
        _, preds = torch.max(px, dim=1)
        accuracy = torch.mean((preds != lx).float())
        print(f'[+] val {log_loss:.5f} {accuracy:.3f}')

        if log_loss < min_loss:
            torch.save(model.state_dict(), 'densenet161_15130.pth')
            print(f'[+] val score improved from {min_loss:.5f} to {log_loss:.5f}. Saved!')
            min_loss = log_loss
            patience = 0
        else:
            patience += 1

'''
模块六.构建densenet201网络结构
'''

def get_model():
    print('[+] loading model... ', end='', flush=True)
    model = models.densenet201_finetune(NB_CLASSES)
    if use_gpu:
        model.cuda()
    print('done')
    return model


def train():
    train_dataset = FurnitureDataset('train', transform=preprocess_with_augmentation(normalize_torch, IMAGE_SIZE))
    val_dataset = FurnitureDataset('val', transform=preprocess(normalize_torch, IMAGE_SIZE))
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=8,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)
    validation_data_loader = DataLoader(dataset=val_dataset, num_workers=1,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)

    model = get_model()

    criterion = nn.CrossEntropyLoss().cuda()

    nb_learnable_params = sum(p.numel() for p in model.fresh_params())
    print(f'[+] nb learnable params {nb_learnable_params}')

    lx, px = utils.predict(model, validation_data_loader)
    min_loss = criterion(Variable(px), Variable(lx)).data[0]

    lr = 0
    patience = 0
    for epoch in range(20):
        print(f'epoch {epoch}')
        if epoch == 1:
            lr = 0.00003
            print(f'[+] set lr={lr}')
        if patience == 2:
            patience = 0
            model.load_state_dict(torch.load('densenet201_15755.pth'))
            lr = lr / 10
            print(f'[+] set lr={lr}')
        if epoch == 0:
            lr = 0.001
            print(f'[+] set lr={lr}')
            optimizer = torch.optim.Adam(model.fresh_params(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

        running_loss = RunningMean()
        running_score = RunningMean()

        model.train()
        pbar = tqdm(training_data_loader, total=len(training_data_loader))
        for inputs, labels in pbar:
            batch_size = inputs.size(0)

            inputs = Variable(inputs)
            labels = Variable(labels)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, dim=1)

            loss = criterion(outputs, labels)
            running_loss.update(loss.data[0], 1)
            running_score.update(torch.sum(preds != labels.data), batch_size)

            loss.backward()
            optimizer.step()

            pbar.set_description(f'{running_loss.value:.5f} {running_score.value:.3f}')
        print(f'[+] epoch {epoch} {running_loss.value:.5f} {running_score.value:.3f}')

        lx, px = utils.predict(model, validation_data_loader)
        log_loss = criterion(Variable(px), Variable(lx))
        log_loss = log_loss.data[0]
        _, preds = torch.max(px, dim=1)
        accuracy = torch.mean((preds != lx).float())
        print(f'[+] val {log_loss:.5f} {accuracy:.3f}')

        if log_loss < min_loss:
            torch.save(model.state_dict(), 'densenet201_15755.pth')
            print(f'[+] val score improved from {min_loss:.5f} to {log_loss:.5f}. Saved!')
            min_loss = log_loss
            patience = 0
        else:
            patience += 1


'''
模块7.构建inceptionresnetv2网络
'''
def get_model():
    print('[+] loading model... ', end='', flush=True)
    model = models.inceptionresnetv2_finetune(NB_CLASSES)
    if use_gpu:
        model.cuda()
    print('done')
    return model


def train():
    train_dataset = FurnitureDataset('train', transform=preprocess_with_augmentation(normalize_05, IMAGE_SIZE))
    val_dataset = FurnitureDataset('val', transform=preprocess(normalize_05, IMAGE_SIZE))
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=8,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)
    validation_data_loader = DataLoader(dataset=val_dataset, num_workers=1,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)

    model = get_model()

    criterion = nn.CrossEntropyLoss().cuda()

    nb_learnable_params = sum(p.numel() for p in model.fresh_params())
    print(f'[+] nb learnable params {nb_learnable_params}')

    lx, px = utils.predict(model, validation_data_loader)
    min_loss = criterion(Variable(px), Variable(lx)).data[0]

    lr = 0
    patience = 0
    for epoch in range(20):
        print(f'epoch {epoch}')
        if epoch == 1:
            lr = 0.00003
            print(f'[+] set lr={lr}')
        if patience == 2:
            patience = 0
            model.load_state_dict(torch.load('inceptionresnetv2_049438.pth'))
            lr = lr / 10
            print(f'[+] set lr={lr}')
        if epoch == 0:
            lr = 0.001
            print(f'[+] set lr={lr}')
            optimizer = torch.optim.Adam(model.fresh_params(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

        running_loss = RunningMean()
        running_score = RunningMean()

        model.train()
        pbar = tqdm(training_data_loader, total=len(training_data_loader))
        for inputs, labels in pbar:
            batch_size = inputs.size(0)

            inputs = Variable(inputs)
            labels = Variable(labels)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, dim=1)

            loss = criterion(outputs, labels)
            running_loss.update(loss.data[0], 1)
            running_score.update(torch.sum(preds != labels.data), batch_size)

            loss.backward()
            optimizer.step()

            pbar.set_description(f'{running_loss.value:.5f} {running_score.value:.3f}')
        print(f'[+] epoch {epoch} {running_loss.value:.5f} {running_score.value:.3f}')

        lx, px = utils.predict(model, validation_data_loader)
        log_loss = criterion(Variable(px), Variable(lx))
        log_loss = log_loss.data[0]
        _, preds = torch.max(px, dim=1)
        accuracy = torch.mean((preds != lx).float())
        print(f'[+] val {log_loss:.5f} {accuracy:.3f}')

        if log_loss < min_loss:
            torch.save(model.state_dict(), 'inceptionresnetv2_049438.pth')
            print(f'[+] val score improved from {min_loss:.5f} to {log_loss:.5f}. Saved!')
            min_loss = log_loss
            patience = 0
        else:
            patience += 1



'''
模块8.构建 xception 网络
'''

'''
神经网络设计原则:
(1)避免采用带有瓶颈的层，尤其在网络结构开始的时候。对于一个前向传播网络，
可以将其看作一个有向五环图（从输入到分类或者回归层）。将输入与输出进行分开，
都能导致大量的信息从分开处流失。一般的情况下，特征图的大小从输入到输出应该缓慢下降。
理论上讲，很多信息不能通过特征维数来得到，比如相关性结构。维度智能代表一些粗略的信息。

(2)高维度能够很更容易在网络的局部进行处理。在卷积网络结构中，增加非线性能够使得更多的特征解耦合。
从而使的网络训练速度更快。

(3)空间聚合能够在低维嵌入进行，然而不会带来任何表达能的减弱。例如，在进行3x3的卷积时，可以在空间聚合之前，对输入进行降维，而不会带来严重的影响。原因：如果采用空间聚合，则相邻的位置的信息具有强相关性，即使进行了降维，也不会带来太多的损失，并且维数的降低，也能够加速网络学习。

(4)平衡网络的宽度和深度。最优的网络可以通过平衡每一个阶段的滤波器的个数和网络的深度达到。
网络的宽度和深度的增加可以使的网络达到了一个更高的效果。
但是，最优的网络结构都是通过同时来提升网络的宽度和深度，但是也需要考虑计算资源的分配。

'''

'''
第三个模型:Xception
Xception是基于Inception的网络基础上而成的。
Inception是先1*1卷积降维再用3*3卷积提取特征，这样计算量减少，中间用了俩个relu，并在适当时机用batch_norm正则化。
Xception是用3*3卷积提取特征，再用1*1卷积融合。这个叫做: depthwise separable convlution
Xception同时用了resnet的残差连接。
'''

'''
我们的下载文件中。
Pil image.open可以通过其内容了解GIF和PNG等文件格式。但是脚本的作用是将所有文件转换为JPEG。
具体看: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
'''

def get_model():
    print('[+] loading model... ', end='', flush=True)
    model = models.xception_finetune(NB_CLASSES)
    if use_gpu:
        model.cuda()
    print('done')
    return model


def train():
    train_dataset = FurnitureDataset('train', transform=preprocess_with_augmentation(normalize_05, IMAGE_SIZE))
    val_dataset = FurnitureDataset('val', transform=preprocess(normalize_05, IMAGE_SIZE))
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=8,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)
    validation_data_loader = DataLoader(dataset=val_dataset, num_workers=1,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)

    model = get_model()

    criterion = nn.CrossEntropyLoss().cuda()

    nb_learnable_params = sum(p.numel() for p in model.fresh_params())
    print(f'[+] nb learnable params {nb_learnable_params}')

    lx, px = utils.predict(model, validation_data_loader)
    min_loss = criterion(Variable(px), Variable(lx)).data[0]

    lr = 0
    patience = 0
    for epoch in range(20):
        print(f'epoch {epoch}')
        if epoch == 1:
            lr = 0.00003
            print(f'[+] set lr={lr}')
        if patience == 2:
            patience = 0
            model.load_state_dict(torch.load('xception_053719.pth'))
            lr = lr / 10
            print(f'[+] set lr={lr}')
        if epoch == 0:
            lr = 0.001
            print(f'[+] set lr={lr}')
            optimizer = torch.optim.Adam(model.fresh_params(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

        running_loss = RunningMean()
        running_score = RunningMean()

        model.train()
        pbar = tqdm(training_data_loader, total=len(training_data_loader))
        for inputs, labels in pbar:
            batch_size = inputs.size(0)

            inputs = Variable(inputs)
            labels = Variable(labels)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, dim=1)

            loss = criterion(outputs, labels)
            running_loss.update(loss.data[0], 1)
            running_score.update(torch.sum(preds != labels.data), batch_size)

            loss.backward()
            optimizer.step()

            pbar.set_description(f'{running_loss.value:.5f} {running_score.value:.3f}')
        print(f'[+] epoch {epoch} {running_loss.value:.5f} {running_score.value:.3f}')

        lx, px = utils.predict(model, validation_data_loader)
        log_loss = criterion(Variable(px), Variable(lx))
        log_loss = log_loss.data[0]
        _, preds = torch.max(px, dim=1)
        accuracy = torch.mean((preds != lx).float())
        print(f'[+] val {log_loss:.5f} {accuracy:.3f}')

        if log_loss < min_loss:
            torch.save(model.state_dict(), 'xception_053719.pth')
            print(f'[+] val score improved from {min_loss:.5f} to {log_loss:.5f}. Saved!')
            min_loss = log_loss
            patience = 0
        else:
            patience += 1


'''
模块9. 预测
'''

def get_model(model_class):
    print('[+] loading model... ', end='', flush=True)
    model = model_class(NB_CLASSES)
    if use_gpu:
        model.cuda()
    print('done')
    return model


def predict(model_name, model_class, weight_pth, image_size, normalize):
    print(f'[+] predict {model_name}')
    model = get_model(model_class)
    model.load_state_dict(torch.load(weight_pth))
    model.eval()

    tta_preprocess = [preprocess(normalize, image_size), preprocess_hflip(normalize, image_size)]
    tta_preprocess += make_transforms([transforms.Resize((image_size + 20, image_size + 20))],
                                      [transforms.ToTensor(), normalize],
                                      five_crops(image_size))
    tta_preprocess += make_transforms([transforms.Resize((image_size + 20, image_size + 20))],
                                      [HorizontalFlip(), transforms.ToTensor(), normalize],
                                      five_crops(image_size))
    print(f'[+] tta size: {len(tta_preprocess)}')

    data_loaders = []
    for transform in tta_preprocess:
        test_dataset = FurnitureDataset('test', transform=transform)
        data_loader = DataLoader(dataset=test_dataset, num_workers=1,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
        data_loaders.append(data_loader)

    lx, px = utils.predict_tta(model, data_loaders)
    data = {
        'lx': lx.cpu(),
        'px': px.cpu(),
    }
    torch.save(data, f'{model_name}_test_prediction.pth')

    data_loaders = []
    for transform in tta_preprocess:
        test_dataset = FurnitureDataset('val', transform=transform)
        data_loader = DataLoader(dataset=test_dataset, num_workers=1,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
        data_loaders.append(data_loader)

    lx, px = utils.predict_tta(model, data_loaders)
    data = {
        'lx': lx.cpu(),
        'px': px.cpu(),
    }
    torch.save(data, f'{model_name}_val_prediction.pth')


def predict_all():
    predict("inceptionv4", models.inceptionv4_finetune, 'inception4_052382.pth', 299, normalize_05)
    predict("densenet161", models.densenet161_finetune, 'densenet161_15130.pth', 224, normalize_torch)
    predict("densenet201", models.densenet201_finetune, 'densenet201_15755.pth', 224, normalize_torch)
    predict("inceptionresnetv2", models.inceptionresnetv2_finetune, 'inceptionresnetv2_049438.pth', 299, normalize_05)
    predict("xception", models.xception_finetune, 'xception_053719.pth', 299, normalize_05)

'''
提交文件
'''

train_json = json.load(open('data/train.json'))
train_df_0 = pd.DataFrame(train_json['annotations'])
train_df_1 = pd.DataFrame(train_json['images'])
train_df = pd.merge(train_df_0, train_df_1)

paths = []
val_pred1 = torch.load('inceptionv4_val_prediction.pth', map_location={'cuda:0': 'cpu'})
val_pred2 = torch.load('densenet161_val_prediction.pth', map_location={'cuda:0': 'cpu'})
val_pred3 = torch.load('densenet201_val_prediction.pth', map_location={'cuda:0': 'cpu'})
val_pred4 = torch.load('inceptionresnetv2_val_prediction.pth', map_location={'cuda:0': 'cpu'})
val_pred5 = torch.load('xception_val_prediction.pth', map_location={'cuda:0': 'cpu'})
val_prob = F.softmax(Variable(torch.cat((
    val_pred1['px'],
    val_pred2['px'],
    val_pred3['px'],
    val_pred4['px'],
    val_pred5['px'],
), dim=2)), dim=1).data.numpy()

val_prob = gmean(val_prob, axis=2)
val_pred = np.argmax(val_prob, axis=1)


def calibrate_prob(positive_prob_train, positive_prob_test, prob):
    return (positive_prob_test * prob) / (positive_prob_test * prob + positive_prob_train * (1 - prob))


def calibrate_probs(prob):
    nb_train = train_df.shape[0]
    for class_ in range(128):
        nb_positive_train = ((train_df.label_id - 1) == class_).sum()

        positive_prob_train = nb_positive_train / nb_train
        positive_prob_test = 1 / 128  # balanced class distribution
        for i in range(prob.shape[0]):
            old_p = prob[i, class_]
            new_p = calibrate_prob(positive_prob_train, positive_prob_test, old_p)
            prob[i, class_] = new_p


calibrate_probs(val_prob)

# Score before calibration
np.mean(val_pred != val_pred1['lx'].numpy())

# Score after calibration
np.mean(np.argmax(val_prob, axis=1) != val_pred1['lx'].numpy())

f, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, sharey=True, figsize=(15, 5))
pd.DataFrame(train_df.label_id.value_counts()).hist(ax=ax0)
ax0.set_title('Train')

pd.Series(val_pred1['lx'].numpy()).value_counts().hist(ax=ax1)
ax1.set_title('Val Ideal')

pd.Series(val_pred).value_counts().hist(ax=ax2)
ax2.set_title('Raw val prediction')

pd.DataFrame(np.argmax(val_prob, axis=1))[0].value_counts().hist(ax=ax3)
ax3.set_title('Calibrated val prediction')
f;

test_pred1 = torch.load('inceptionv4_test_prediction.pth', map_location={'cuda:0': 'cpu'})
test_pred2 = torch.load('densenet161_test_prediction.pth', map_location={'cuda:0': 'cpu'})
test_pred3 = torch.load('densenet201_test_prediction.pth', map_location={'cuda:0': 'cpu'})
test_pred4 = torch.load('inceptionresnetv2_test_prediction.pth', map_location={'cuda:0': 'cpu'})
test_pred5 = torch.load('xception_test_prediction.pth', map_location={'cuda:0': 'cpu'})

test_prob = F.softmax(Variable(torch.cat((
    test_pred1['px'],
    test_pred2['px'],
    test_pred3['px'],
    test_pred4['px'],
    test_pred5['px'],
), dim=2)), dim=1).data.numpy()

test_prob = gmean(test_prob, axis=2)
test_prob.shape

test_predicted = np.argmax(test_prob, axis=1) + 1

calibrate_probs(test_prob)
calibrated_predicted = np.argmax(test_prob, axis=1) + 1

f, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(15, 5))
pd.Series(test_predicted).value_counts().hist(ax=ax0)
ax0.set_title('Before')

pd.Series(calibrated_predicted).value_counts().hist(ax=ax1)
ax1.set_title('After')
f;

test_dataset = FurnitureDataset('test', transform=preprocess)
sx = pd.read_csv('data/sample_submission_randomlabel.csv')
sx.loc[sx.id.isin(test_dataset.data.image_id), 'predicted'] = calibrated_predicted
sx.to_csv('sx_calibrated.csv', index=False)
