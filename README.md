# CDMA
official code for: Semi-supervised Pathological Image Segmentation via Cross Distillation of Multiple Attentions. MICCAI 2023, early accept [arxiv](https://arxiv.org/abs/2305.18830).
And the extension is published on the [Pattern Recognition](https://www.sciencedirect.com/science/article/pii/S0031320324002437) 2024.

### Overall Framework
There are three branches based on different attention mechanisms and two losses in our framework
![overall](https://github.com/HiLab-git/CDMA/blob/main/pics/overall2.png)

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

You can get data lists in ```data/digestpath```
### Citation
```
@inproceedings{zhong2023semi,
  title={Semi-supervised Pathological Image Segmentation via Cross Distillation of Multiple Attentions},
  author={Zhong, Lanfeng and Liao, Xin and Zhang, Shaoting and Wang, Guotai},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={570--579},
  year={2023},
  organization={Springer}
}

@article{zhong2024semi,
  title={Semi-supervised pathological image segmentation via cross distillation of multiple attentions and Seg-CAM consistency},
  author={Zhong, Lanfeng and Luo, Xiangde and Liao, Xin and Zhang, Shaoting and Wang, Guotai},
  journal={Pattern Recognition},
  pages={110492},
  year={2024},
  publisher={Elsevier}
}
```

### Acknowledgement
The code of semi-supervised learning framework is borrowed from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)

