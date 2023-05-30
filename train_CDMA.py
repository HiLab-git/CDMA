import argparse
import os
import random
import time
import sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
join = os.path.join
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import monai
import torch.optim as optim
from dataloaders.dataset import get_train_loader, get_val_loader, get_val_WSI_loader
from monai.data import decollate_batch, PILReader
from monai.inferers import sliding_window_inference
from utils.Metrics import DiceMetric
from utils.losses import DiceLoss, KDLoss, entropy_loss
import logging
from core.networks import MTNet
from tensorboardX import SummaryWriter


def get_arguments():
    parser = argparse.ArgumentParser(description="CDMA Pytorch implementation on Digest Path 2019 ")
    parser.add_argument("--dataset_root", type=str,
                        default="", help="training images")
    parser.add_argument("--batch_size", type=int,
                        default=16, help="Train batch size")
    parser.add_argument("--labeled_bs", type=int, default=8)
    parser.add_argument("--num_class", type=int,
                        default=2, help="Train class num")
    parser.add_argument("--input_size", default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--gpu", nargs="+", type=int)
    parser.add_argument("--save_folder", default="model")
    parser.add_argument("--num_workers", default=6)
    parser.add_argument("--max_epoch", default=150, type=int)
    parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
    parser.add_argument("--portion", default=5, type=int)
    return parser.parse_args()

def get_files(data_root):
    new_file = []
    img_names = os.listdir(data_root+'images')
    for img_name in img_names:
        image_root = data_root+'images/'+img_name
        label_root = data_root+'labels/'+img_name[:-4]+'_mask.png'
        new_sample = {'img': image_root, 'label': label_root}
        new_file.append(new_sample)
    return new_file

def get_deeplab(args, ema=False):
    model = MTNet("resnet50", num_classes=args.num_class, use_group_norm=True)
    model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()
    param_groups = model.module.get_parameter_groups(None)
    optimizer = optim.SGD(
        [
            {"params": param_groups[0], "lr": args.lr},
            {"params": param_groups[1], "lr": 2 * args.lr},
            {"params": param_groups[2], "lr": 10 * args.lr},
            {"params": param_groups[3], "lr": 20 * args.lr},
        ],
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    if ema:
        for param in model.module.parameters():
                param.detach_()
    return model, optimizer

def get_files(data_root):
    new_file = []
    img_names = os.listdir(data_root+'images')
    for img_name in img_names:
        image_root = data_root+'images/'+img_name
        label_root = data_root+'labels/'+img_name[:-4]+'_mask.png'
        new_sample = {'img':image_root, 'label':label_root}
        new_file.append(new_sample)
    return new_file

def train(model, train_loader, optimizer, iter_num, epoch):
    model.train()
    kd_loss = KDLoss(T=10)
    epoch_loss_sup = 0
    epoch_loss_en = 0
    epoch_loss_cross = 0
    epoch_loss_unsup = 0
    for batch_data in train_loader:
        batch_names = batch_data['img_meta_dict']['filename_or_obj']
        labeled_names = labeled_names + batch_names[:args.labeled_bs]    
        unlabeled_names = unlabeled_names + batch_names[args.labeled_bs:]

        inputs, labels = batch_data["img"].float().cuda(), batch_data["label"].cuda()

        outputs1, outputs2, outputs3 = model(inputs)

        outputs1_soft = torch.softmax(outputs1, dim=1)
        outputs2_soft = torch.softmax(outputs2, dim=1)
        outputs3_soft = torch.softmax(outputs3, dim=1)

        loss_sup = 0.5*dice_loss(outputs1_soft[:args.labeled_bs], labels[:args.labeled_bs])+0.5*F.cross_entropy(outputs1[:args.labeled_bs], labels[:args.labeled_bs,0,:,:].long()) + \
            0.5*dice_loss(outputs2_soft[:args.labeled_bs], labels[:args.labeled_bs])+0.5*F.cross_entropy(outputs2[:args.labeled_bs], labels[:args.labeled_bs,0,:,:].long()) + \
            0.5*dice_loss(outputs3_soft[:args.labeled_bs], labels[:args.labeled_bs])+0.5*F.cross_entropy(outputs3[:args.labeled_bs], labels[:args.labeled_bs,0,:,:].long())

        loss_sup = loss_sup/3

        # entropy loss
        outputs_avg_soft = (outputs1_soft+outputs2_soft+outputs3_soft)/3
        en_loss = entropy_loss(outputs_avg_soft, C=2)

        cross_loss1 = kd_loss(outputs1.permute(0, 2, 3, 1).reshape(-1, 2),outputs2.detach().permute(0, 2, 3, 1).reshape(-1, 2)) + \
            kd_loss(outputs1.permute(0, 2, 3, 1).reshape(-1, 2),outputs3.detach().permute(0, 2, 3, 1).reshape(-1, 2))
        cross_loss2 = kd_loss(outputs2.permute(0, 2, 3, 1).reshape(-1, 2),outputs1.detach().permute(0, 2, 3, 1).reshape(-1, 2)) + \
            kd_loss(outputs2.permute(0, 2, 3, 1).reshape(-1, 2),outputs3.detach().permute(0, 2, 3, 1).reshape(-1, 2))
        cross_loss3 = kd_loss(outputs3.permute(0, 2, 3, 1).reshape(-1, 2),outputs1.detach().permute(0, 2, 3, 1).reshape(-1, 2)) + \
            kd_loss(outputs3.permute(0, 2, 3, 1).reshape(-1, 2),outputs2.detach().permute(0, 2, 3, 1).reshape(-1, 2))

        cross_consist = (cross_loss1 + cross_loss2 + cross_loss3)/3

        # overall function
        cross_weight = args.consistency
        en_weight = args.consistency
        en_loss = en_weight * en_loss
        cross_loss = cross_weight*cross_consist
        if epoch < 15:
            consistency_loss = torch.tensor((0,)).cuda()
        else:
            consistency_loss = cross_loss + en_loss
        loss = loss_sup + consistency_loss

        iter_num += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for param_group in optimizer.param_groups:
            lr_ = param_group['lr']

        epoch_loss_unsup += consistency_loss.item()
        epoch_loss_sup += loss_sup.item()
        epoch_loss_en += en_loss.item()
        epoch_loss_cross += cross_loss.item()
    print('sup loss:', epoch_loss_sup/len(train_loader), "unsup loss", epoch_loss_unsup/len(train_loader),'en_loss:', epoch_loss_en/(len(train_loader)),\
        'cdma cross_loss:', epoch_loss_cross/(len(train_loader)))
    return epoch_loss_sup/len(train_loader), lr_

def validate(model, val_loader):
    model.eval()
    dice_metric = DiceMetric(num_class=args.num_class)
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data["img"].cuda(), val_data["label"].cuda()
            val_outputs1, _, _ = model(val_images)
            val_outputs = val_outputs1
            dice_metric.add_batch(val_outputs,val_labels[:,0,:,:])
    dice_value = dice_metric.compute_dice()
    print(dice_value)
    return dice_value

def validate_WSI(model, val_loader, overlap=0.25, save_folder=None, save_csv=None):
    model.eval()
    dice_metric = DiceMetric(num_class=args.num_class)
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data["img"].cuda(), val_data["label"].cuda()
            val_outputs = sliding_window_inference(val_images, [args.input_size, args.input_size], 4, model, overlap=overlap)
            preds = val_outputs
            dice_metric.add_batch(preds, val_labels[:, 0, :, :])
            # save pics
            batch_names = val_data['img_meta_dict']['filename_or_obj']
            sample_name = batch_names[0]
            sample_name = sample_name.split('/')[-1]

            val_numpy = preds[0].permute(0, 2, 1).cpu().numpy()
            val_pred = val_numpy.argmax(0)
            val_pred = np.array(val_pred*255, dtype=np.uint8)

            if save_folder:
                if not os.path.exists(save_folder):
                    os.mkdir(save_folder)
                Image.fromarray(val_pred).save(save_folder+sample_name[:-4]+'.png')
    dice_value = dice_metric.compute_dice(save=save_csv)
    print(dice_value)
    return dice_value

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    args = get_arguments()
    portion = args.portion
    writer = SummaryWriter(f'tensorborad/cdma/deeplab/{portion}_portion')
    logging.basicConfig(level=logging.INFO, filename=f'log/cdma_{portion}.txt')

    # set rand seed
    setup_seed(1)

    labeled_data_root = f'/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-{portion}-patch/'
    all_data_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-100-patch/'
    val_data_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-val-patch/'
    test_data_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-test-patch/'

    labeled_files = get_files(labeled_data_root)
    all_data_files = get_files(all_data_root)

    np.random.shuffle(labeled_files)

    labeled_num = len(labeled_files)
    all_data_num = len(all_data_files)

    labeled_data_img_names = []
    for i in range(labeled_num):
        img_path = labeled_files[i]['img']
        img_name = img_path.split('/')
        img_name = img_name[-1]
        labeled_data_img_names.append(img_name)

    labeled_idxs = []
    unlabeled_idxs = []
    for i in range(all_data_num):
        img_path = all_data_files[i]['img']
        img_name = img_path.split('/')
        img_name = img_name[-1]
        if img_name in labeled_data_img_names:
            labeled_idxs.append(i)
        else:
            unlabeled_idxs.append(i)
    
    logging.info(f'labeled:{labeled_num},unlabeled:{all_data_num-labeled_num}')
    print(f'labeled:{labeled_num},unlabeled:{all_data_num-labeled_num}')

    val_files = get_files(val_data_root)

    logging.info(f'training files:{all_data_num}, valid files:{len(val_files)}')
    print(f'training files:{all_data_num}, valid files:{len(val_files)}')

    train_loader = get_train_loader(args, all_data_files, labeled_idxs, unlabeled_idxs)
    val_loader = get_val_loader(args, val_files)

    dice_loss = DiceLoss(n_classes=args.num_class)

    max_epoch = args.max_epoch
    iter_num = 0
    print(f'max_epoch:{max_epoch}')
    logging.info(f'max_epoch:{max_epoch}')
    max_dice = 0

    # set gpu
    torch.cuda.set_device(args.gpu[0])
    # get model
    model, optimizer = get_deeplab(args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch//4,max_epoch//2,max_epoch*3//4], gamma=0.5)

    labeled_names = []
    unlabeled_names = []
    for epoch in range(max_epoch):
        t0 = time.time()
        train_loss, cur_lr = train(model, train_loader, optimizer, iter_num, epoch)

        t1 = time.time()
        val_dice = validate(model, val_loader)
        t2 = time.time()
        scheduler.step()

        iter_num = (epoch+1)*len(train_loader)

        print("training/validation time: {0:.2f}s/{1:.2f}s".format(t1 - t0, t2 - t1))

        if val_dice.mean() > max_dice:
            max_dice = val_dice.mean()
            best_epoch = epoch+1
            print(f'cur_best dice:{max_dice}')
            torch.save(model.module.state_dict(), f'model/cdma_{portion}_best.pth')
    # # test
    print('------------test-------------')
    save_folder = f'test_results/{portion}_cdma/'
    test_WSI_data_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-test/'
    test_WSI_files = get_files(test_WSI_data_root)
    test_WSI_loader = get_val_WSI_loader(test_WSI_files, args)
    test_model = MTNet("resnet50", num_classes=args.num_class, use_group_norm=True, train=False).cuda()
    test_model.eval()
    ckpt = torch.load(f'model/cmda_{portion}_best.pth', map_location="cpu")
    test_model.load_state_dict(ckpt, strict=True)
    test_dice_WSI = validate_WSI(test_model, test_WSI_loader, overlap=0.25, save_folder=save_folder, save_csv=f'results_csv/cdma_{portion}.csv')
    logging.info('test dice {0:.4f}'.format(test_dice_WSI))