import argparse
import os
from random import seed, shuffle
import random
import time
import sys
from PIL import Image
from test import count_label_unlabel
Image.MAX_IMAGE_PIXELS = None
join = os.path.join
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import monai
import torch.optim as optim
from dataloaders.dataset import get_train_loader, get_val_loader
from train_baseline import get_files_redistribution, get_val_WSI_loader
from monai.data import decollate_batch, PILReader
from monai.inferers import sliding_window_inference
from utils.Metrics import DiceMetric
from utils.losses import DiceLoss, KDLoss, entropy_loss
import logging
import csv
from networks.unet import UNet
from core.networks import MTNet_Plus
from tensorboardX import SummaryWriter
from medpy import metric
from torch import nn


def get_arguments():
    parser = argparse.ArgumentParser(description="Digest Path 2019 Pytorch implementation")
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
    parser.add_argument("--max_epoch", default=40, type=int)

    parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float, default=30.0, help='consistency_rampup')

    parser.add_argument("--portion", default=5, type=int)
    return parser.parse_args()


def get_deeplab(args, ema=False):
    model = MTNet_Plus(model_name="resnet50", num_classes=args.num_class, use_group_norm=True)
    model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()
    param_groups = model.get_parameter_groups(None)
    optimizer = optim.SGD(
        [
            {"params": param_groups[0], "lr": args.lr},
            {"params": param_groups[1], "lr": 2 * args.lr},
            {"params": param_groups[2], "lr": 10 * args.lr},
            {"params": param_groups[3], "lr": 20 * args.lr},
        ],
        # params=model.module.parameters(),
        # lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    if ema:
        for param in model.parameters():
                param.detach_()
    return model, optimizer

def update_ema_variables(model, ema_model):
    # Use the true average until the exponential average is more correct
    alpha = 0.99
    # print('alpha:',alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def get_files(data_root):
    new_file = []
    img_names = os.listdir(data_root+'images')
    for img_name in img_names:
        image_root = data_root+'images/'+img_name
        label_root = data_root+'labels/'+img_name[:-4]+'_mask.png'
        new_sample = {'img':image_root, 'label':label_root}
        new_file.append(new_sample)
    return new_file

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

def train(model, ema_model, train_loader, optimizer, iter_num, epoch, labeled_names, unlabeled_names):
    model.train()
    kd_loss = KDLoss(T=10)
    epoch_loss = 0
    scaler = torch.cuda.amp.GradScaler()
    for batch_data in train_loader:
        batch_names = batch_data['img_meta_dict']['filename_or_obj']
        labeled_names = labeled_names + batch_names[:args.labeled_bs]
        unlabeled_names = unlabeled_names + batch_names[args.labeled_bs:]

        inputs, labels = batch_data["img"].float().cuda(), batch_data["label"].cuda()
        unlabeled_inputs = inputs[args.labeled_bs:]

        # generate the classification model
        logits_labels = torch.zeros((inputs.shape[0])).long().cuda()
        for i in range(inputs.shape[0]):
            seg_label = labels[i].cpu().numpy()
            if np.max(seg_label) == 1:
                logits_labels[i] = 1
        logits_labels_onehot = F.one_hot(logits_labels, num_classes=2)

        with torch.cuda.amp.autocast():
            outputs1, outputs2, outputs3, logits, cams = model(inputs)

            # print(outputs.shape, logits.shape)
            outputs1_soft = torch.softmax(outputs1, dim=1)
            outputs2_soft = torch.softmax(outputs2, dim=1)
            outputs3_soft = torch.softmax(outputs3, dim=1)
            outputs_soft_avg = (outputs1_soft+outputs2_soft+outputs3_soft)/3
            outputs_avg = (outputs1+outputs2+outputs3)/3
            logits_soft = torch.softmax(logits, dim=1)
            # print(cams.shape)

            loss_sup_seg = (0.5*dice_loss(outputs1_soft[:args.labeled_bs], labels[:args.labeled_bs])+0.5*F.cross_entropy(outputs1[:args.labeled_bs], labels[:args.labeled_bs,0,:,:].long()) + \
                0.5*dice_loss(outputs2_soft[:args.labeled_bs], labels[:args.labeled_bs])+0.5*F.cross_entropy(outputs2[:args.labeled_bs], labels[:args.labeled_bs,0,:,:].long()) + \
                0.5*dice_loss(outputs3_soft[:args.labeled_bs], labels[:args.labeled_bs])+0.5*F.cross_entropy(outputs3[:args.labeled_bs], labels[:args.labeled_bs,0,:,:].long()))/3        

            loss_cls = F.binary_cross_entropy_with_logits(logits[:args.labeled_bs], logits_labels_onehot[:args.labeled_bs].float())

            if epoch < 15:
                consistency_weight = 0
            else:
                consistency_weight = args.consistency

            cross_loss1 = kd_loss(outputs1.permute(0, 2, 3, 1).reshape(-1, 2),outputs2.detach().permute(0, 2, 3, 1).reshape(-1, 2)) + \
                kd_loss(outputs1.permute(0, 2, 3, 1).reshape(-1, 2),outputs3.detach().permute(0, 2, 3, 1).reshape(-1, 2))
            cross_loss2 = kd_loss(outputs2.permute(0, 2, 3, 1).reshape(-1, 2),outputs1.detach().permute(0, 2, 3, 1).reshape(-1, 2)) + \
                kd_loss(outputs2.permute(0, 2, 3, 1).reshape(-1, 2),outputs3.detach().permute(0, 2, 3, 1).reshape(-1, 2))
            cross_loss3 = kd_loss(outputs3.permute(0, 2, 3, 1).reshape(-1, 2),outputs1.detach().permute(0, 2, 3, 1).reshape(-1, 2)) + \
                kd_loss(outputs3.permute(0, 2, 3, 1).reshape(-1, 2),outputs2.detach().permute(0, 2, 3, 1).reshape(-1, 2))
            consistency_loss_cross = (cross_loss1+cross_loss2+cross_loss3)/3

            consistency_task = torch.mean((outputs_soft_avg-cams)**2)

            """entropy minimization for segmentation branch, or classification branch"""
            en_loss = entropy_loss(outputs_soft_avg, C=2)

            consistency_loss = consistency_loss_cross + consistency_task + en_loss
            loss_sup = loss_sup_seg + 0.1*loss_cls
            loss = loss_sup + consistency_weight * consistency_loss

            update_ema_variables(model, ema_model)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        for param_group in optimizer.param_groups:
            lr_ = param_group['lr']

        epoch_loss += loss.item()
    print('train_loss:', epoch_loss/len(train_loader), 'multi-task consistency_loss:', consistency_weight * consistency_loss.item(), \
        'cross consistency loss:', consistency_loss_cross.item(), 'task loss:', consistency_task.item(), 'entropy loss:', en_loss.item())
    return epoch_loss/len(train_loader), lr_, labeled_names, unlabeled_names

def validate(model, val_loader, save_img=False, save_heatmap=False):
    acc, count = 0, 0
    model.eval()
    dice = 0
    dice_count = 0
    # dice_metric = DiceMetric(num_class=args.num_class)
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data["img"].cuda(), val_data["label"].cuda()
            # generate the classification model
            logits_labels = torch.zeros((val_labels.shape[0],))
            for i in range(val_labels.shape[0]):
                seg_label = val_labels[i].cpu().numpy()
                if np.max(seg_label) == 1:
                    logits_labels[i] = 1
            val_outputs, val_outputs2, val_outputs3, val_logits, val_cams = model(val_images)
            # softmax
            # val_outputs_soft = torch.softmax(val_outputs, dim=1)

            # no softmax
            val_outputs = F.relu(val_outputs)
            val_outputs_soft = val_outputs / (F.adaptive_max_pool2d(val_outputs, 1) + 1e-5)

            val_outputs2 = F.relu(val_outputs2)
            val_outputs2_soft = val_outputs2 / (F.adaptive_max_pool2d(val_outputs2, 1) + 1e-5)

            val_outputs3 = F.relu(val_outputs3)
            val_outputs3_soft = val_outputs3 / (F.adaptive_max_pool2d(val_outputs3, 1) + 1e-5)

            logits_labels = logits_labels.numpy()
            val_logits = val_logits.cpu().numpy()
            val_logits = val_logits.argmax(1)
            acc += np.sum(logits_labels == val_logits)
            count += val_logits.shape[0]

            for i in range(len(logits_labels)):
                if logits_labels[i] == 1:
                    # val_outputs = (torch.softmax(val_outputs, dim=1)+val_cams)/2
                    val_output_i = val_outputs[i].argmax(0).cpu().numpy().astype(np.uint8)
                    val_label_i = val_labels[i].cpu().numpy().astype(np.uint8)
                    # print(val_output_i.max(), val_label_i.max())
                    dice += metric.dc(val_output_i, val_label_i)
                    # print(dice)
                    dice_count += 1
                    # dice_metric.add_batch(val_outputs[i], val_labels[:, 0, :, :])

                if save_img:
                    batch_names = val_data['img_meta_dict']['filename_or_obj']
                    sample_name = batch_names[i]
                    sample_name = sample_name.split('/')[-1]

                    val_numpy = val_outputs_soft[i].permute(0, 2, 1).cpu().numpy()
                    val_pred = val_numpy[1]
                    val_pred = np.array(val_pred*255, dtype=np.uint8)
                    Image.fromarray(val_pred).save('test_results_patch_hard/branch1/'+sample_name[:-4]+'.png')

                    val_numpy = val_outputs2_soft[i].permute(0, 2, 1).cpu().numpy()
                    val_pred = val_numpy[1]
                    val_pred = np.array(val_pred*255, dtype=np.uint8)
                    Image.fromarray(val_pred).save('test_results_patch_hard/branch2/'+sample_name[:-4]+'.png')

                    val_numpy = val_outputs3_soft[i].permute(0, 2, 1).cpu().numpy()
                    val_pred = val_numpy[1]
                    val_pred = np.array(val_pred*255, dtype=np.uint8)

                    Image.fromarray(val_pred).save('test_results_patch_hard/branch3/'+sample_name[:-4]+'.png')

                    val_numpy = ((val_outputs_soft[i]+val_outputs2_soft[i]+val_outputs3_soft[i])/3).permute(0, 2, 1).cpu().numpy()
                    val_pred = val_numpy[1]
                    val_pred = np.array(val_pred*255, dtype=np.uint8)
                    Image.fromarray(val_pred).save('test_results_patch/avg/'+sample_name[:-4]+'.png')

                if save_heatmap:
                    batch_names = val_data['img_meta_dict']['filename_or_obj']
                    sample_name = batch_names[i]
                    sample_name = sample_name.split('/')[-1]

                    val_numpy = val_cams[i].permute(0, 2, 1).cpu().numpy()
                    val_pred = val_numpy[1]
                    # print(np.max(val_pred), np.min(val_pred))
                    val_pred = np.array(val_pred*255, dtype=np.uint8)
                    Image.fromarray(val_pred).save('test_results_patch/branch1_cams/'+sample_name[:-4]+'.png')

    print(dice/(dice_count+1e-5), acc/count)
    return dice/(dice_count+1e-5)*100

def validate_WSI(model, val_loader, save_folder):
    model.eval()
    dice, count = 0, 0
    dice_list = []
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data["img"].cuda(), val_data["label"].cuda()
            val_outputs,val_outputs2,val_outputs3,val_cams = sliding_window_inference(val_images, [args.input_size, args.input_size], 4, model, overlap=0.25)
            val_label = val_labels[0].cpu().numpy().astype(np.uint8)
            # val_outputs = (val_outputs+val_outputs2+val_outputs3)/3
            
            # for segmentation, use threshold
            val_outputs_soft = torch.softmax(val_outputs, dim=1)

            val_pred = val_outputs[0].argmax(0).cpu().numpy().astype(np.uint8)

            dice_cur = metric.dc(val_pred, val_label)
            dice += dice_cur
            dice_list.append(dice_cur)
            count += 1
            # save pics
            batch_names = val_data['img_meta_dict']['filename_or_obj']
            sample_name = batch_names[0]
            sample_name = sample_name.split('/')[-1]

            val_numpy = val_outputs[0].permute(0, 2, 1).cpu().numpy()
            val_pred = val_numpy.argmax(0)
            val_pred = np.array(val_pred*255, dtype=np.uint8)

            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            Image.fromarray(val_pred).save(save_folder+sample_name[:-4]+'.png')
    print(dice/count)
    return dice/count*100

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
    l = logging.getLogger(__name__)
    fileHandler = logging.FileHandler(f'log/cdma_plus_{portion}.log', mode='a')
    l.setLevel(logging.INFO)
    l.addHandler(fileHandler)

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
    
    l.info(f'labeled:{labeled_num},unlabeled:{all_data_num-labeled_num}')
    print(f'labeled:{labeled_num},unlabeled:{all_data_num-labeled_num}')

    val_pos, val_neg = get_files_redistribution(val_data_root)
    val_files = val_pos

    test_pos, test_neg = get_files_redistribution(test_data_root)
    test_files = test_pos + test_neg
    
    l.info(f'training files:{all_data_num}, valid files:{len(val_files)}')
    print(f'training files:{all_data_num}, valid files:{len(val_files)}')

    train_loader = get_train_loader(args, all_data_files, labeled_idxs, unlabeled_idxs)
    val_loader = get_val_loader(args, val_files)
    test_loader = get_val_loader(args, test_files)

    dice_loss = DiceLoss(n_classes=args.num_class)

    max_epoch = 150

    iter_num = 0
    print(f'max_epoch:{max_epoch}')
    l.info(f'max_epoch:{max_epoch}')
    max_dice = 0

    # set gpu
    torch.cuda.set_device(args.gpu[0])
    # get model
    model, optimizer = get_deeplab(args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch//4, max_epoch//2, max_epoch*3//4], gamma=0.5)

    model_ema, _ = get_deeplab(args, ema=True)

    labeled_names = []
    unlabeled_names = []
    for epoch in range(max_epoch):
        t0 = time.time()
        train_loss, cur_lr, labeled_names, unlabeled_names = train(model, model_ema, train_loader, optimizer, iter_num, epoch, labeled_names, unlabeled_names)
        t1 = time.time()
        val_dice = validate(model, val_loader)
        t2 = time.time()
        scheduler.step()

        iter_num = (epoch+1)*len(train_loader)

        print("training/validation time: {0:.2f}s/{1:.2f}s".format(t1 - t0, t2 - t1))

        if val_dice > max_dice:
            max_dice = val_dice
            best_epoch = epoch+1
            print(f'cur_best dice:{max_dice}')
            torch.save(model.state_dict(), f'model/cdma_plus_{portion}_best.pth')
    # # test
    print('------------test-------------')
    save_folder = f'test_results/{portion}_multi_task_lanfz/'
    test_WSI_data_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-test/'
    test_WSI_files = get_files(test_WSI_data_root)
    test_WSI_loader = get_val_WSI_loader(test_WSI_files, args)

    # test in WSI
    test_model = MTNet_Plus(model_name="resnet50", num_classes=args.num_class, use_group_norm=True, train=False).cuda()
    test_model.eval()
    ckpt = torch.load(f'model/cmda_plus_{portion}_best.pth', map_location="cpu")
    test_model.load_state_dict(ckpt, strict=True)
    validate(test_model, test_loader, save_img=True, save_heatmap=True)
    val_dice = validate_WSI(test_model, test_WSI_loader, save_folder)
    l.info('test dice {0:.4f}'.format(val_dice))

