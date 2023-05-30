from nntplib import NNTPDataError
from random import shuffle
import numpy as np
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# neg_root = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-neg/'
# neg_names = os.listdir(neg_root)

# for neg_name in neg_names:
#     neg_path = neg_root+neg_name
#     neg_img = Image.open(neg_path)
#     mask_neg = np.zeros((neg_img.size[1],neg_img.size[0]), dtype=np.uint8)
#     Image.fromarray(mask_neg).save('/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-neg-mask/'+neg_name[:-4]+'_mask.jpg')

image_dir= '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-pos-v2/'
label_dir = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-pos-v2/'
# means edg to inter, no two means edg to background

label_re_dir = '/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-pos-v2/'
label_names = os.listdir(label_dir)
print(len(label_names))
for label_name in label_names:
    if 'mask' in label_name:
        label_path = label_dir + label_name
        # image_path = image_dir + label_name[:-9] + '.jpg'
        label_img = np.array(Image.open(label_path),dtype=np.uint8)
        print(np.unique(label_img))
        # if len(np.unique(label_img))!=1 and len(np.unique(label_img))!=2:
        #     print(np.unique(label_img),label_name)
        #     os.remove(label_path)
        #     os.remove(image_path)
        # label_img[label_img>128] = 255
        # label_img[label_img<=128] = 0
        # label_img[label_img==255] = 1
        # print(np.unique(label_img))
        # Image.fromarray(label_img).save(label_re_dir+label_name[:-4]+'.png')