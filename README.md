# Structured Binary Neural Networks for Image Recognition

created by [Bohan Zhuang](https://sites.google.com/view/bohanzhuang)


***If you use this code in your research, please cite our paper:***

```
@inproceedings{zhuang2019structured,
  title={Structured Binary Neural Networks for Accurate Image Classification and Semantic Segmentation},
  author={Zhuang, Bohan and Shen, Chunhua and Tan, Mingkui and Liu, Lingqiao and Reid, Ian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={413--422},
  year={2019}
}
```

## This is a simple implementation for imagenet classification.


pretrain.py:     pretrain using Tanh()

finetune.py:     finetune the binary model 

model.py:    define the model (with softgates)

model_no_softgate.py:        define the model (without softgates)

model_no_scales_softgate.py:   define the model (without learnt scales and softgates)

new_layers.py:      define necessary quantization functions

utils.py:	 define auxiliary functions

resnet18_without_softgates.pth.tar:	 pretrained model for ResNet-18 without softgates, [google drive](https://drive.google.com/open?id=1b6osR-T-9swwUEG7AXG1hIWB7GoYho5R)

resnet18_with_softgates.pth.tar:    pretrained model for ResNet-18 with softgates, [google drive](https://drive.google.com/open?id=1YP94KWujdnCy5sUis-giHWeZ3n8o2pAD)

resnet34_without_softgates.pth.tar:  pretrained model for ResNet-34 without softgates, [google drive](https://drive.google.com/open?id=122IsP7ysPQLK0akFloZq9297TLQQsKL8)

resnet34_with_softgates.pth.tar:   pretrained model for ResNet-34 with softgates, [google drive](https://drive.google.com/open?id=1ISnvB7PNz0jgDqlPFpyqcWuo32njh_fN)



**You can safely remove scales in the code and replace them with the averaging operation.**


resnet18_without_scales_softgates.pth.tar:   pretrained model for ResNet-18 without softgates and learnt scales, [google drive](https://drive.google.com/open?id=1Z9r9FK1yrAFyYgmZ45MUv7QXEEZUwG9Q)



## Semantic segmentation and object detection

Please refer to [semantic segmentation](https://bitbucket.org/jingruixiaozhuang/group-net-semantic-segmentation/src/master/) and [object detection](https://bitbucket.org/jingruixiaozhuang/group-net-object-detection/src/master/).

## Copyright

Copyright (c) Bohan Zhuang. 2019

** This code is for non-commercial purposes only. For commerical purposes,
please contact Chunhua Shen <chhshen@gmail.com> **

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

This program is distributed in the hope that it will be useful,https://bitbucket.org/jingruixiaozhuang/group_net/src/master/README.md?mode=edit&spa=0&at=master&fileviewer=file-view-default#
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
