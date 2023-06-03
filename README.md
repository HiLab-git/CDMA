# CDMA
offical code for: Semi-supervised Pathological Image Segmentation via Cross Distillation of Multiple Attentions. MICCAI 2023, provisional accept [arxiv](https://arxiv.org/abs/2305.18830).

### usage
First, split the dataset and crop WSIs into patches.
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

### Acknowledgement
The code of semi-supervise learning framework is borrowed from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)
