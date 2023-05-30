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


def checkBlank(patch):
    patch = np.array(patch.convert("RGB"))
    m = patch.mean()
    if 200 <= m <= 255:
        return False
    else:
        return True


def filter_names(mask_root, img_names):
    new_names = []
    for img_name in img_names:
        img = Image.open(mask_root+img_name)
        if img.size[0] < 8000 and img.size[1] < 8000:
            new_names.append(img_name)
    return new_names


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
        source_image = source_root+name
        source_label = source_root+name[:-4]+'_mask.png'
        shutil.copyfile(source_image, target_image_root+name)
        shutil.copyfile(source_label, target_label_root+name[:-4]+'_mask.png')


if __name__ == "__main__":
    np.random.seed(66)

    all_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-pos-v2/'
    all_names = os.listdir(all_root)
    all_names = filter_mask(all_names)
    all_names = filter_names(all_root, all_names)
    np.random.shuffle(all_names)
    print(len(all_names))

    train_names_5 = all_names[:5]
    train_names_15 = all_names[:10]
    train_names_30 = all_names[:20]
    train_names_150 = all_names[:100]
    val_names = all_names[100:110]
    test_names = all_names[110:130]
    # train_names = all_names[:25*7]
    # val_names = all_names[25*7:25*8]
    # test_names = all_names[25*8:]

    # # move_file(all_root, '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train/', train_names)
    # move_file('/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-pos-v2/',
    #           '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-2/', train_names_2)
    move_file('/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-pos-v2/',
              '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-5/', train_names_5)
    move_file('/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-pos-v2/',
              '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-10/', train_names_15)
    move_file('/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-pos-v2/',
              '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-20/', train_names_30)
    move_file('/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-pos-v2/',
              '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-100/', train_names_150)
    # move_file('/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-pos-v2/',
    #           '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-100/', train_names_100)
    move_file('/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-pos-v2/',
              '/mnt/data2/lanfz/datasets/digestpath2019/tissue-val/', val_names)
    move_file('/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-pos-v2/',
              '/mnt/data2/lanfz/datasets/digestpath2019/tissue-test/', test_names)
    # move_file('/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-pos/', '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-pos-test/', test_names)

    # source_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-pos-v1/'
    # target_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-pos-v2/'
    # img_names = os.listdir(source_root)
    # for img_name in img_names:
    #     if 'mask' not in img_name:
    #         shutil.copyfile(source_root+img_name, target_root+img_name)
