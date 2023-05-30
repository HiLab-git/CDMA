from PIL import Image
import os
import numpy as np


# [15998 15998-3272], [2404, 2404-592]
img_names = os.listdir('/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-patch/labels')
count = 0
print(len(img_names))
for img_name in img_names:
    label = np.array(Image.open('/mnt/data2/lanfz/datasets/digestpath2019/tissue-train-patch/labels/'+img_name))
    if len(np.unique(label)) == 1:
        count += 1
    #     os.remove('/mnt/data2/lanfz/datasets/digestpath2019/tissue-val-patch/images/'+img_name[:-9]+'.jpg')
    #     os.remove('/mnt/data2/lanfz/datasets/digestpath2019/tissue-val-patch/labels/'+img_name)
    # if count == 900:
    #     break
print(len(img_names), count)
