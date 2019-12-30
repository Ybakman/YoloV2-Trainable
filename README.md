# YoloV2-Trainable
Trainable Tiny-YoloV2 implementation on Julia by Knet framework

The program loads pre-trained weights to layers except the last layer. It initalizes last layer (detection layer) randomly. It freezes other layers and trains last layer only by adam optimizer.

first download pre-trained weights by:
```
$ wget https://pjreddie.com/media/files/yolov2-tiny-voc.weights
```
If you want to train data and see accuracy on Voc Dataset 2007, download the dataset by:
```
$ wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
  tar xf VOCtrainval_06-Nov-2007.tar
```
The repo has trained model which is trained.jld2. If you want to see accuracy for this model run:
```
$ julia YoloTrain.jl accuracy
```
If you want to display the image by using trained.jld2. Run:
```
$ julia YoloTrain.jl loadDisplay
```
To train model, run:
```
$ julia YoloTrain.jl train --batch_size 32 --epochs 20
```
Batch_size = 32 and 20 epochs is suggested. When it is trained, it saves trained model as trained1.jld2. If you want to use this model for accuracy and display, simply use the argument:
```
$ --choose true
```
## Example Input and Output
Here is an example of input and output:

INPUT:
<p align="center">
  <img src="example.jpg" width="416" height="416">
</p> 

OUTPUT:
<p align="center">
  <img src="outexample.jpg" width="416" height="416">
</p> 

## References

https://fairyonice.github.io/Part_4_Object_Detection_with_Yolo_using_VOC_2012_data_loss.html
https://github.com/experiencor/keras-yolo2

## To-Do List

-It will become a more generalized version to train different datasets.


