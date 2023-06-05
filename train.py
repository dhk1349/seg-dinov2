import argparse
import time
import datetime
import os
import shutil
import sys
from tqdm import tqdm
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import transforms
from models.dino2 import DINO2SEG
from utils.mscoco import COCOSegmentation
from utils.segmentationMetric import *
from utils.vis import decode_segmap 

from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')

    parser.add_argument('--base-size', type=int, default=580,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=518,
                        help='crop image size')

    # training hyper params
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')

    # checkpoint and log
    parser.add_argument('--save-dir', default='./ckpt',
                        help='Directory for saving checkpoint models')

    args = parser.parse_args()
    return args

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        trainset = COCOSegmentation('/mnt/2tb/mscoco/', transform=input_transform)
        valset = COCOSegmentation('/mnt/2tb/mscoco/', 'val', transform=input_transform)
        
        self.train_loader = data.DataLoader(dataset=trainset, batch_size=args.batch_size,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=valset, batch_size=args.batch_size,
                                          pin_memory=True)

        self.model = nn.DataParallel(DINO2SEG(len(trainset.classes)).to(self.device))
    
        # create criterion
        a =  1/42
        b = (1-a)/20
        weights = torch.tensor([a] + [b for _ in range(20)]).to('cuda')
        self.criterion = nn.CrossEntropyLoss(weight=weights) 

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                         lr=args.lr)

        self.metric = SegmentationMetric(len(trainset.classes))
        self.best_pred = -1

    def train(self):
        iteration = 0
        avg_loss = 0
        for i in range(args.epochs):
            self.validation(iteration, i)
            self.model.train()
            for images, targets, _ in self.train_loader:
                iteration = iteration + 1
                # self.lr_scheduler.step()

                images = images.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(images)
                pred = torch.max(outputs, 1).indices
                loss = self.criterion(outputs, targets)

                loss = torch.mean(loss)
            
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss

                
                if iteration % 100 == 0:
                    print(f"e {i} |{iteration} it: {avg_loss.item()/100}")
                    writer.add_scalar('training loss', avg_loss.item()/100, iteration)
                    avg_loss = 0

                # if iteration % 1000 == 1:
                #     pred = decode_segmap(pred[0].cpu().data.numpy())
                #     gt = decode_segmap(targets[0].cpu().data.numpy())

                #     pred = torch.from_numpy(pred).permute(2, 0, 1)
                #     gt = torch.from_numpy(gt).permute(2, 0, 1)
                #     writer.add_image("pred", pred, iteration)
                #     writer.add_image("gt", gt, iteration)


    def validation(self, it, e):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        torch.cuda.empty_cache() 

        self.model.eval()
        _preds = []
        _targets = []
        print("Evaluating")
        for image, target, _ in tqdm(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = self.model(image)
            self.metric.update(outputs, target)
            pixAcc, mIoU = self.metric.get()

            pred = torch.max(outputs, 1).indices
            for i in range(pred.shape[0]):
                if len(_preds)<64:
                    _preds.append(torchvision.transforms.ToTensor()(decode_segmap(pred[i].cpu().data.numpy())))
                    _targets.append(torchvision.transforms.ToTensor()(decode_segmap(target[i].cpu().data.numpy())))

        _preds = torchvision.utils.make_grid(_preds, nrow=8)
        _targets = torchvision.utils.make_grid(_targets, nrow=8)

        new_pred = (pixAcc + mIoU) / 2
        print(f"pixel acc: {pixAcc}\nmIoU: {mIoU}")
        writer.add_scalar('validation pixAcc', pixAcc, it)
        writer.add_scalar('validation mIoU', mIoU, it)
        writer.add_image("gt", _targets, it)
        writer.add_image("pred", _preds, it)

        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred

        save_checkpoint(self.model, self.args, is_best)


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"dinov2_mscoco.pth"
    filename = os.path.join(directory, filename)

    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = 'dinov2_mscoco_best_model.pth'
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

    # reference maskrcnn-benchmark
    args.device = "cuda"
    writer = SummaryWriter()
    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
