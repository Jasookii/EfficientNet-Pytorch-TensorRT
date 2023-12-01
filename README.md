# EfficientNet-Pytorch-TensorRT

## Pytorch

1. Install `efficientnet_pytorch`
```
pip install efficientnet_pytorch
```

2. Prepare dataset as `ImageNet1K`

   1. prepare dataset as follows

      ```
      dataset/
      ├── class1
      │   ├── class1_img1.JPEG
      │   ├── class1_img2.JPEG
      │   └── ...
      ├── class2
      └── ...
      ```

   2. split dataset as ImageNet1K

      ```
      python dataset_split.py
      ```

      ```
      dataset/
      ├── train/
      │   ├── class1
      │   │   ├── class1_img1.JPEG
      │   │   ├── class1_img2.JPEG
      │   │   └── ...
      │		└── ...
      ├── val/
      │   ├── class1
      │   │   ├── class1_test1.JPEG
      │   │   ├── class1_test2.JPEG
      │   │   └── ...
      │		└── ...
      ```


3. Train&Test EfficientNet, (Note that the number of classes)

```
python main_img.py
python main_matrix.py
```

Matrix CheckPoints: https://drive.google.com/file/d/1hMDgn0Cs4QElBYz3RBIsCgrborv5HVG6/view?usp=drive_link

IMG CheckPoints: https://drive.google.com/file/d/1V1CegkkA_6ln-2XVP_6qffQy2kkRA6yc/view?usp=drive_link

## TRT

1. gennerate `.wts` file

```
python gen_wts.py
```

2. build

```
mkdir build
cd build
cmake ..
make
```
3. serialize model to engine

```
./efficientnet -s [.wts] [.engine] [b0-signal b0 b1 b2 b3 ... b7]  // serialize model to engine file
```
4. deserialize and do infer

```
./efficientnet -d [.engine] [b0-signal b0 b1 b2 b3 ... b7] [img-path]  // deserialize engine and do inference
```

## TRT Server

Serve

Client

## Implementation

For the TensorRT implementation, you can refer to [tensorrtx](https://github.com/wang-xinyu/tensorrtx/tree/master)

For the Pytorch implementation, you can refer to [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
