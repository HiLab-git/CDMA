# CDMA
official code for: Semi-supervised Pathological Image Segmentation via Cross Distillation of Multiple Attentions. MICCAI 2023, provisional accept [arxiv](https://arxiv.org/abs/2305.18830).

### Overall Framework
There are three branches based on different attention mechanisms and two losses in our framework
![overall](https://github.com/HiLab-git/CDMA/blob/main/pics/overall.png)

### usage
First, split the dataset into train, val and test sets, then crop WSIs into patches for computational feasibility.
```
python utils.move_file.py
python slide_window.py
```

Then, just use the ```run.sh``` script to run the code.
```
sh run.sh
```

### Data Acquisition
The DigestPath dataset can be downloaded in: [DigestPath](https://digestpath2019.grand-challenge.org/)

The dataset dir is like this after splitting and cropping:
```
digestpath2019
-----tissue-train-100
-----tissue-train-100-patch
-----tissue-train-5
-----tissue-train-5-patch
-----tissue-val
-----tissue-val-patch
-----tissue-test
```

### Citation
```
@article{zhong2023semi,
  title={Semi-supervised Pathological Image Segmentation via Cross Distillation of Multiple Attentions},
  author={Zhong, Lanfeng and Liao, Xin and Zhang, Shaoting and Wang, Guotai},
  journal={arXiv preprint arXiv:2305.18830},
  year={2023}
}
```

### Acknowledgement
The code of semi-supervised learning framework is borrowed from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)

