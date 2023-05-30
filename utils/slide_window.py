from move_file import move_file
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

def slide_window_inference(image, model):
    pass

def slide_crop(img_dataroot,mask_dataroot,img_save_root,mask_save_root):
    # clean
    if not os.path.exists(img_save_root):
        os.mkdir(img_save_root)
        os.mkdir(mask_save_root)
    else:
        shutil.rmtree(img_save_root)
        shutil.rmtree(mask_save_root)
        os.mkdir(img_save_root)
        os.mkdir(mask_save_root)

    img_names = os.listdir(img_dataroot)
    img_names = filter_mask(img_names)

    for img_name in img_names[:]:
        # images
        img = Image.open(img_dataroot + img_name).convert("RGB")
        mask = Image.open(mask_dataroot + img_name[:-4]+'_mask.png')

        # params
        img_shape = img.size
        img_dim = len(img_shape)
        window_size = [256, 256]
        window_stride = [256, 256]

        for d in range(img_dim):
            if (window_size[d] is None) or window_size[d] > img_shape[d]:
                window_size[d] = img_shape[d]
            if (window_stride[d] is None) or window_stride[d] > window_size[d]:
                window_stride[d] = window_size[d]

        crop_start_list = []
        for w in range(0, img_shape[-1], window_stride[-1]):
            w_min = min(w, img_shape[-1] - window_size[-1])
            for h in range(0, img_shape[-2], window_stride[-2]):
                h_min = min(h, img_shape[-2] - window_size[-2])
                if img_dim == 2:
                    crop_start_list.append([h_min, w_min])
                else:
                    for d in range(0, img_shape[0], window_stride[0]):
                        d_min = min(d, img_shape[0] - window_size[0])
                        crop_start_list.append([d_min, h_min, w_min])
        i = 0
        for c0 in crop_start_list:
            c1 = [c0[d] + window_size[d] for d in range(img_dim)]
            img_patch = img.crop((c0[0], c0[1], c1[0], c1[1]))
            # here, we check the blank patch
            if checkBlank(img_patch):
                img_patch.save(
                    img_save_root
                    + img_name[:-4]
                    + f"_{i}.jpg"
                )
                mask_patch = mask.crop((c0[0], c0[1], c1[0], c1[1]))
                mask_patch.save(
                    mask_save_root
                    + img_name[:-4]
                    + f"_{i}_mask.png"
                )
                i += 1
    print("done!")

if __name__ == "__main__":

    # img_dataroot = "/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-2/images/"
    # mask_dataroot = "/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-2/labels_v2/"

    # img_save_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-2-patch/images/'
    # mask_save_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-2-patch/labels_v2/'

    # slide_crop(img_dataroot, mask_dataroot, img_save_root, mask_save_root)

    img_dataroot = "/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-50/images/"
    mask_dataroot = "/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-50/labels/"

    img_save_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-50-patch/images/'
    mask_save_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-50-patch/labels/'

    slide_crop(img_dataroot, mask_dataroot, img_save_root, mask_save_root)

    # img_dataroot = "/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-10/images/"
    # mask_dataroot = "/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-10/labels_v2/"

    # img_save_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-10-patch/images/'
    # mask_save_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-10-patch/labels_v2/'

    # slide_crop(img_dataroot, mask_dataroot, img_save_root, mask_save_root)

    # img_dataroot = "/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-20/images/"
    # mask_dataroot = "/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-20/labels_v2/"

    # img_save_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-20-patch/images/'
    # mask_save_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-20-patch/labels_v2/'

    # slide_crop(img_dataroot, mask_dataroot, img_save_root, mask_save_root)

    # img_dataroot = "/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-100/images/"
    # mask_dataroot = "/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-100/labels_v2/"

    # img_save_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-100-patch/images/'
    # mask_save_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-100-patch/labels_v2/'

    # slide_crop(img_dataroot, mask_dataroot, img_save_root, mask_save_root)

    # img_dataroot = "/mnt/data2/lanfz/datasets/digestpath2019/tissue-val/images/"
    # mask_dataroot = "/mnt/data2/lanfz/datasets/digestpath2019/tissue-val/labels_v2/"

    # img_save_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-val-patch/images/'
    # mask_save_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-val-patch/labels_v2/'

    # slide_crop(img_dataroot, mask_dataroot, img_save_root, mask_save_root)

    # img_dataroot = "/mnt/data2/lanfz/datasets/digestpath2019/tissue-test/images/"
    # mask_dataroot = "/mnt/data2/lanfz/datasets/digestpath2019/tissue-test/labels/"

    # img_save_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-test-patch/images/'
    # mask_save_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-test-patch/labels/'

    # slide_crop(img_dataroot, mask_dataroot, img_save_root, mask_save_root)