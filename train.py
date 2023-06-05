import argparse
import time
import datetime
import os
import shutil
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from models.dino2 import DINO2SEG
from utils.mscoco import COCOSegmentation
from utils.segmentationMetric import *
from utils.vis import decode_segmap 
from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='fcn',
                        choices=['fcn32s', 'fcn16s', 'fcn8s', 'fcn', 'psp', 'deeplabv3', 
                            'deeplabv3_plus', 'danet', 'denseaspp', 'bisenet', 'encnet', 
                            'dunet', 'icnet', 'enet', 'ocnet', 'psanet', 'cgnet', 'espnet', 
                            'lednet', 'dfanet','swnet'],
                        help='model name (default: fcn32s)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet18', 'resnet50', 'resnet101', 'resnet152', 
                            'densenet121', 'densenet161', 'densenet169', 'densenet201'],
                        help='backbone name (default: vgg16)')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        choices=['pascal_voc', 'pascal_aug', 'ade20k', 'citys', 'sbu'],
                        help='dataset name (default: pascal_voc)')
    parser.add_argument('--base-size', type=int, default=580,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=518,
                        help='crop image size')
    parser.add_argument('--workers', '-j', type=int, default=4,
                        metavar='N', help='dataloader threads')
    # training hyper params
    parser.add_argument('--jpu', action='store_true', default=False,
                        help='JPU')
    parser.add_argument('--use-ohem', type=bool, default=False,
                        help='OHEM Loss for cityscapes dataset')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--warmup-iters', type=int, default=0,
                        help='warmup iters')
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3,
                        help='lr = warmup_factor * lr')
    parser.add_argument('--warmup-method', type=str, default='linear',
                        help='method of warmup')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='~/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'pascal_aug': 80,
            'pascal_voc': 50,
            'pcontext': 80,
            'ade20k': 160,
            'citys': 120,
            'sbu': 160,
        }
        args.epochs = epoches[args.dataset.lower()]
    if args.lr is None:
        lrs = {
            'coco': 0.004,
            'pascal_aug': 0.001,
            'pascal_voc': 0.0001,
            'pcontext': 0.001,
            'ade20k': 0.01,
            'citys': 0.01,
            'sbu': 0.001,
        }
        args.lr = lrs[args.dataset.lower()] / 8 * args.batch_size
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
        """
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        train_dataset = get_segmentation_dataset(args.dataset, split='train', mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)
        args.iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)
        args.max_iters = args.epochs * args.iters_per_epoch

        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iters)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)
        """
        
        self.train_loader = data.DataLoader(dataset=trainset, batch_size=args.batch_size,
                                            num_workers=args.workers,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=valset, batch_size=args.batch_size,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
        #                                     aux=args.aux, jpu=args.jpu, norm_layer=BatchNorm2d).to(self.device)
        self.model = nn.DataParallel(DINO2SEG(21).to(self.device))
    
        # create criterion
        a =  1/42
        b = (1-a)/20
        weights = torch.tensor([a] + [b for _ in range(20)]).to('cuda')
        self.criterion = nn.CrossEntropyLoss(weight=weights) # get_segmentation_loss(args.model, use_ohem=args.use_ohem, aux=args.aux,
                              #                  aux_weight=args.aux_weight, ignore_index=-1).to(self.device)

        # optimizer, for model just includes pretrained, head and auxlayer
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay)
        """
        # lr scheduling
        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                         max_iters=args.max_iters,
                                         power=0.9,
                                         warmup_factor=args.warmup_factor,
                                         warmup_iters=args.warmup_iters,
                                         warmup_method=args.warmup_method)


        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)

        self.best_pred = 0.0
        """
    def train(self):
        # epochs, max_iters = self.args.epochs, self.args.max_iters
        # log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.args.iters_per_epoch
        # save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        start_time = time.time()
        # logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()
        # for iteration, (images, targets, _) in enumerate(self.train_loader):
        iteration = 0
        avg_loss = 0
        for i in range(20):
            for images, targets, _ in self.train_loader:
                iteration = iteration + 1
                # self.lr_scheduler.step()

                images = images.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(images)
                pred = torch.max(outputs, 1).indices
                loss = self.criterion(outputs, targets)
                # losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                # loss_dict_reduced = reduce_loss_dict(loss_dict)
                # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                loss = torch.mean(loss)
            
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss
                # eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
                # eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                if iteration % 100 == 0:
                    print(f"e {i} |{iteration} it: {avg_loss.item()/100}")
                    writer.add_scalar('training loss', avg_loss.item()/100, iteration)
                    avg_loss = 0

                if iteration % 1000 == 1:
                    pred = decode_segmap(pred[0].cpu().data.numpy())
                    gt = decode_segmap(targets[0].cpu().data.numpy())

                    pred = torch.from_numpy(pred).permute(2, 0, 1)
                    gt = torch.from_numpy(gt).permute(2, 0, 1)
                    writer.add_image("pred", pred, iteration)
                    writer.add_image("gt", gt, iteration)
                    # writer.add_image("org", images[0].cpu().data.numpy(), iteration)
            # if iteration % save_per_iters == 0 and save_to_disk:
                # save_checkpoint(self.model, self.args, is_best=False)
            # if not self.args.skip_val and iteration % val_per_iters == 0:
            #     self.validation()
            #     self.model.train()
        
        """
        save_checkpoint(self.model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))
        """
    def validation(self):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)
            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        save_checkpoint(self.model, self.args, is_best)
        synchronize()


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}.pth'.format(args.model, args.backbone, args.dataset)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_{}_{}_best_model.pth'.format(args.model, args.backbone, args.dataset)
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
