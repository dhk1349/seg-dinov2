# seg-dinov2
Fine-tuning dino v2 for semantic segmentation task on MSCOCO.


## Implementation Detail

### Backbone
* dinov2-vitb/14 as backbone 

### Head
* Linear layer + conv layer 

* variation of linear tuning 
(refer to section 7.4 of [Dinov2](https://arxiv.org/abs/2304.07193))

### Data preparation code for segmentation task
* [Semantic Segmentation on PyTorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)

### Trained on 
RTX3090\*2 with batch size of 8. 


## Prediction

<img src="./imgs/pred.png">
<p style="text-align:center">Prediction</p>

<img src="./imgs/gt.png">
<p style="text-align:center">GT</p>
