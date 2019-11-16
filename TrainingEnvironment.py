from torch.utils.data import Dataset, DataLoader
from utils import *

import torchvision
import torch.nn as nn
from torch import optim
from termcolor import colored
# extra tools for data augmentation, this is not mine...
import utils.custom_transforms as custom
# from Networks.StyleNet import StyleAugmentation

import torch.nn.functional as F


def Load_loaders_and_optimization(self):
    # loaders
    transformations_set_aug, transformations_setting = [], []
    if self.conf.style is not None:
        print("Stylization is being used...")
        transformations_set_aug.extend([transforms.Resize(256, interpolation=2),
                                        transforms.ToTensor(),
                                        custom.Stylization(layer=self.conf.style[0], # (0.25 , 0.5, False, False, 3., 0.) 
                                                    alpha=self.conf.style[1],
                                                    remove_stochastic=self.conf.style[2],
                                                    prob=self.conf.style[3],
                                                    pseudo1=self.conf.style[4],
                                                    Noise=self.conf.style[5],
                                                    std=self.conf.style[6],
                                                    mean=self.conf.style[7],),
                                        transforms.ToPILImage(),
                                        ])# Optimize this section

    if self.conf.resize is not None: # For resize images...
        print("Resize is being used...")
        transformations_set_aug.extend([transforms.Resize(self.conf.resize, interpolation=2),])
        transformations_setting.extend([transforms.Resize(self.conf.resize, interpolation=2),])

    if self.conf.augmentation:
        print("Data augmentation is being used...")
        if self.conf.rot: transformations_set_aug.extend([ transforms.RandomRotation(self.conf.rot, resample=False, expand=False, center=None), ])
        if self.conf.hflip: transformations_set_aug.extend([ transforms.RandomHorizontalFlip(p=self.conf.hflip), ])
        if self.conf.vflip: transformations_set_aug.extend([ transforms.RandomVerticalFlip(p=self.conf.vflip), ])
        if self.conf.color: transformations_set_aug.extend([ transforms.ColorJitter(brightness=self.conf.color[0], contrast=self.conf.color[1], saturation=self.conf.color[2], hue=self.conf.color[3]) ])
        if self.conf.trans: transformations_set_aug.extend([ transforms.RandomAffine(0, translate=self.conf.trans[0], scale=self.conf.trans[1], shear=self.conf.trans[2], resample=False, fillcolor=0), ])
        transformations_set_aug.extend([transforms.ToTensor(),])
        if self.conf.cutout: transformations_set_aug.extend([ custom.CutoutDefault(length=self.conf.cutout), ])
        if self.conf.erase: transformations_set_aug.extend([ custom.RandomErasing(probability=self.conf.erase[0], sh=self.conf.erase[1], r1=self.conf.erase[2]), ])
    else:
        transformations_set_aug.extend([ transforms.ToTensor(), ])
    transformations_set_aug.extend([ transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    transformations_setting.extend([ transforms.ToTensor(), 
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
    
    self.conf.dataset = self.conf.dataset.upper()
    if self.conf.dataset=="STL10":
        if self.conf.augmentation or self.conf.resize or self.conf.style:
            trainset = torchvision.datasets.STL10(root='./Database/', 
                                                    split='train',  
                                                    transform=transforms.Compose(transformations_set_aug), 
                                                    target_transform=None, 
                                                    download=True)
            testset = torchvision.datasets.STL10(root='./Database/', 
                                                    split='test',
                                                    transform=transforms.Compose(transformations_setting), 
                                                    target_transform=None, 
                                                    download=True)
        else:
            trainset = torchvision.datasets.STL10(root='./Database/', 
                                                    split='train', 
                                                    transform=transforms.Compose(transformations_setting), 
                                                    target_transform=None, 
                                                    download=True)
            testset = torchvision.datasets.STL10(root='./Database/', 
                                                    split='test',
                                                    transform=transforms.Compose(transformations_setting), 
                                                    target_transform=None, 
                                                    download=True)
    self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.conf.train_batch, shuffle=True, num_workers=self.conf.workers)
    self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.conf.test_batch, shuffle=False, num_workers=self.conf.workers)

    # optimization
    self.conf.type_optimizer = self.conf.type_optimizer.lower()
    self.lr = self.conf.lr
    if self.conf.type_optimizer=='sgd':
        self.optimizer = optim.SGD(self.net.parameters(),
                                    lr=self.lr,
                                    momentum=self.conf.momentum,
                                    nesterov=self.conf.nesterov,
                                    weight_decay=self.conf.weight_decay)
    elif self.conf.type_optimizer=='adam':
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.lr,
                                    betas=self.conf.betas,
                                    weight_decay=self.conf.weight_decay)
    else: raise ValueError('NO OPTIMIZER IMPLEMENTATION')

    # Scheduler
    if '/' in self.conf.scheduler:
        v = self.conf.scheduler.split('/')
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.conf.optimizer, mode=v[0], factor=int(v[1]), patience=int(v[2]), threshold=1e-6)
    elif '-' in self.conf.scheduler:
        milestones, gamma = self.conf.scheduler.split('-')
        milestones = [int(i) for i in milestones.split(',')]
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=float(gamma))
    elif 'cos' in self.conf.scheduler:
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.conf.epochs, eta_min=self.conf.lr)
    elif '@' in self.conf.scheduler:
        milestones, power = self.conf.scheduler.split('@')
        milestones = [int(i) for i in milestones.split(',')]
        class Policy_lr:
            def __init__(poly, milestones, max_iter, power=0.9):
                poly.max_iter = max_iter
                print('max iterations: ',poly.max_iter)
                poly.power = power
                if milestones==0: poly.milestones = list(range(0,poly.max_iter))
                else: poly.milestones = milestones
            def step(poly, iter_):
                if iter_ in poly.milestones:
                    base_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = base_lr * ((1 - float(iter_) / poly.max_iter) ** poly.power)
                    print(colored('Previous lr: {0}  ,  lr policy : {1}'.format(base_lr, new_lr),'yellow'))
                    self.optimizer.param_groups[0]['lr'] = new_lr
        self.scheduler = Policy_lr(milestones, self.epochs, float(power))
    else: 
        raise ValueError('No symbols (- /) recognized')

    # loss function
    if self.conf.loss.upper()=='CE': self.criterion = nn.CrossEntropyLoss()
    else: raise ValueError('NO LOSS IMPLEMENTATION')

def training_process(self, epoch):
    epoch_loss = 0
    epoch_acc = 0
    self.net.train()
    # import cv2
    # import pdb
    for it, (imgs, targets) in enumerate(self.trainloader):
        # Image = np.uint8(imgs.permute(0,2,3,1).numpy()*255)
        # batch, Xi, Yi = Image.shape[:3]
        # Xi *= 4
        # Yi *= 4
        # out = np.zeros([Xi*(4), Yi*(batch//4), 3], dtype=np.uint8)
        # for j in range(Image.shape[0]):
        #     print(j)
        #     out[(j//4)*Xi:(j//4+1)*Xi, j%4*Yi:(j%4+1)*Yi] = cv2.resize(Image[j],None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
        # cv2.imwrite('image.png',cv2.cvtColor( out ,cv2.COLOR_BGR2RGB))
        # pdb.set_trace()

        imgs = imgs.cuda(); targets = targets.cuda()
        outputs = self.net(imgs)

        loss = self.criterion(outputs, targets)
        acc = outputs.max(1)[1].eq(targets).float().mean().item()*100
        epoch_loss += loss.item()
        epoch_acc += acc
        print('   {0:.4f} --- loss: {1:.6f} --- acc: {2:.6f}'.format(it/self.N_train, loss.item(), acc), end='\r')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(epoch)
    return epoch_loss, epoch_acc

def evaluation_process(self, epoch): # ADD MORE EVALUTION TYPES
    """Evaluation without the densecrf with the dice coefficient"""
    epoch_loss = 0
    epoch_acc = 0
    self.net.eval()
    with torch.no_grad():
        for it, (imgs, targets) in enumerate(self.testloader):
            imgs = imgs.cuda(); targets = targets.cuda()
            outputs = self.net(imgs)

            loss = self.criterion(outputs, targets)
            acc = outputs.max(1)[1].eq(targets).float().mean().item()*100
            epoch_loss += loss.item()
            epoch_acc += acc
            print('   {0:.4f} --- loss: {1:.6f} --- acc: {2:.6f}'.format(it/self.N_test, loss.item(), acc), end='\r')
    return epoch_loss, epoch_acc