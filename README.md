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
