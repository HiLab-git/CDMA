export CUDA_VISIBLE_DEVICES=0,1,2,3

# MICCAI Version
python train_CDMA.py --gpu 0 --portion 5

# Journal Version
python train_CDMA_Plus.py --gpu 0 --portion 5
