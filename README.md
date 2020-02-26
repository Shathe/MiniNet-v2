
Enhanced implementation of the MiniNet model presented on the [ICRA 2019](https://www.icra2019.org).

[Link to the paper](https://ieeexplore.ieee.org/abstract/document/8793923)

## Requirements
- Python 2.7
- tensorflow
- imgaug
- cv2
- keras
- scipy



## Downloading Datasets and Weights
Go [here](https://drive.google.com/drive/folders/1xdfwU164M7tJVOaqco-tGMBcQcb1r_ml?usp=sharing) for downloading both the datasets and the trained weights reported in the paper.

For a quick example of how to run it, the Camvid dataset and the trained weights are already in the github repository so there is no need for downloading anything if you are not goint to trained on the Cityscapes dataset.

## Testing MiniNet-v2
When testing, apart from reporting the metrics, the resulting output images are saved in a folder called out_dir.
### Camvid
```
python train.py --train 0 --dataset ./Datasets/camvid \
--checkpoint_path ./weights/Mininetv2/camvid_960x720 \
--n_classes 11 --width 960 --height 720 --output_resize_factor 2 
```
Now you can color those outputs to be able to visualize the results.
```
python from_out_to_color_camvid.py --input_dir ./out_dir/Datasets/camvid --output_dir ./out_dir/Datasets/camvid_colored
```

### Cityscapes
First download the dataset [here](https://drive.google.com/drive/folders/1xdfwU164M7tJVOaqco-tGMBcQcb1r_ml?usp=sharing).
```
python train.py --train 0 --dataset ./Datasets/cityscapes --ignore_label 19 \
--checkpoint_path ./weights/Mininetv2/cityscapes_1024x512 \
--n_classes 19 --width 2048 --height 1024 \
--img_resize_factor 1 --output_resize_factor 2 
```
Now you can color those outputs to be able to visualize the results.
```
python from_out_to_color.py --input_dir ./out_dir/Datasets/cityscapes --output_dir ./out_dir/Datasets/cityscapes
```

## Training MiniNet-v2
When training read the Help of each argument if yo want to change it.
The most common problem is memory issues, if you have them, reduce the max_batch_size argument.
### Camvid
```
python train.py --train 1 --dataset ./Datasets/camvid \
--checkpoint_path ./weights/Mininetv2/camvid_new \
--n_classes 11 --ignore_label 11 --width 960 --height 720 \
--max_batch_size 6 --init_lr 1e-3  --min_lr 1e-5 --epochs 1500 \
--output_resize_factor 2 --median_frequency 0.12
```
### Cityscapes
First download the dataset [here](https://drive.google.com/drive/folders/1xdfwU164M7tJVOaqco-tGMBcQcb1r_ml?usp=sharing).
```
python train.py --train 1 --dataset ./Datasets/cityscapes \
--checkpoint_path ./weights/Mininetv2/cityscapes_1024x512 \
--n_classes 19 --ignore_label 19 --width 1024 --height 512 \
--max_batch_size 6 --init_lr 1e-3  --min_lr 1e-5 --epochs 200 \
--output_resize_factor 2 --img_resize_factor 1 --median_frequency 0.12
```

### Execution time and other metrics (Pytorch implementation)

```
python evaluate_time_pytorch.py  --width 1024 --height 512
```

### Execution time and other metrics (Tensorflow implementation)
```
python evaluate_time_tensorflow.py  --width 1024 --height 512
```
### MiniNet-v2-cpu
For executing the small version, just set the parameter --cpu_version to 1 in all the previous scripts.
```
--cpu_version 1
```
Change other parameters like the weights to load,resolution etc... Depending on the set up to execute.
