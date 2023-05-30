from random import shuffle
import numpy as np
import os
from PIL import Image
import shutil
Image.MAX_IMAGE_PIXELS = None

def filter_mask(img_names):
    new_img_name = []
    for img_name in img_names:
        if 'mask' not in img_name:
            new_img_name.append(img_name)
    return new_img_name

def move_file(source_root, target_root, source_names):
    # clean the original file
    if not os.path.exists(target_root + 'images/'):
        os.mkdir(target_root + 'images/')
        os.mkdir(target_root + 'labels/')
    else:
        shutil.rmtree(target_root + 'images/')
        shutil.rmtree(target_root + 'labels/')
        os.mkdir(target_root + 'images/')
        os.mkdir(target_root + 'labels/')
    target_image_root = target_root + 'images/'
    target_label_root = target_root + 'labels/'
    for name in source_names:
        source_image = source_root+'images/'+name
        source_label = source_root+'labels/'+name[:-4]+'_mask.png'
        shutil.copyfile(source_image, target_image_root+name)
        shutil.copyfile(source_label, target_label_root+name[:-4]+'_mask.png')


if __name__ == "__main__":
    np.random.seed(66)

    all_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-100/images/'
    all_names = os.listdir(all_root)
    all_names = filter_mask(all_names)
    print(len(all_names))

    ten_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-10/images/'
    ten_names = os.listdir(ten_root)
    print(len(ten_names))

    ten9_names = list(set(all_names)-set(ten_names))
    print(len(ten9_names))
    ten2_names = ten_names + ten9_names[:10]
    ten5_names = ten_names + ten9_names[:40]

    move_file('/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-100/',
              '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-20/', ten2_names)
    
    move_file('/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-100/',
              '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-50/', ten5_names)

    # ten2_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-20/images/'
    # ten2_names = os.listdir(ten2_root)
    # print(len(ten2_names))

    # train_names_5 = all_names[:5]
    # train_names_15 = all_names[:10]
    # train_names_30 = all_names[:20]
    # train_names_150 = all_names[:100]
    # val_names = all_names[100:110]
    # test_names = all_names[110:130]
