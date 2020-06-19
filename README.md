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

model.py:    define the model with softgates but without learnt scales. The pretrained model is resnet18_without_scales_with_softgates.pth.tar: [MEGA](https://mega.nz/file/qJgzBB6J#ehhRwwZWB9_UkVSBy_tLjZe2iU4m0fbv3R8GuPrY-cM)

model_ablation_1.py:   define the model with softgates and learnt scales. The pretrained model is resnet18_with_scales_with_softgates.pth.tar: [MEGA](https://mega.nz/file/SJo3FJbS#xf3yh5BhobTPaXgWfJzSGQOK_gekOZiruPEnNZHNBrQ)

model_ablation_2.py:  define the model without softgates and learnt scales. The pretrained model is in [MEGA](https://mega.nz/file/XRxH1ZQS#XzPp17O5zJIw8i6SXTtgv1WWAw2MzLEbif8gbshKi7s)

model_ablation_3.py:   define the model without softgates but with learnt scales. The pretrained model is in [MEGA](https://mega.nz/file/jZpByB7Y#_uhLblXKBHb6-vkAKLZ1M_zVPFQzyOlI9VKr1yZbzbs)

new_layers.py:      define necessary quantization functions

utils.py:	 define auxiliary functions


resnet34_without_softgates.pth.tar:  pretrained model for ResNet-34 without softgates but with learnt scales, [MEGA](https://mega.nz/file/iBwzwDDC#L6_nrwiwo6b07vPyOQsR9dOJCUwFqbPbaAaRMdu1v0M)

resnet34_with_softgates.pth.tar:   pretrained model for ResNet-34 with softgates and learnt scales, [MEGA](https://mega.nz/file/edpHkDTY#dRZd7ESma9AjvDpx0jas4PLcZoY7XZhRlC5SjropU-8)




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
