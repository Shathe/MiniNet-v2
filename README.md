
Enahanced implementation of the MiniNet model presented on the [ICRA 2019](https://www.icra2019.org).

[Link to the paper (soon)](https://ieeexplore.ieee.org/abstract/document/8793923)

## Requirements
- Tensorflow

 

## Citing MiniNet

If you find MiniNet useful in your research, please consider citing:
```
@inproceedings{alonso2019Mininet,
  title={Enhancing V-SLAM Keyframe Selection with an Efficient ConvNet for Semantic Analysis},
  author={Alonso, I{\~n}igo and Riazuelo, Luis and Murillo, Ana C},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2019}
}
```


## Training MiniNet-v2
```
python train.py --dataset Datasets/camvid --checkpoint_path models/camvid/ --init_lr 0.001 --batch_size 12 --epochs 500 --n_classes 11 --width 1024 --height 512 --train 1 --mininet_version 2
```


## Testing MiniNet
```
python train.py --train 0 --dataset Datasets/camvid --checkpoint_path models/camvid/ --n_classes 11 --width 512 --height 256  --mininet_version 2
```
