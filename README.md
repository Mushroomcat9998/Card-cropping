## Crop card from background

### Model: U^2 net
#### Torch version:
reference repo: code [here](https://github.com/xuebinqin/U-2-Net)\
training phase: code [here](https://github.com/Mushroomcat9998/U2NET-training)
#### Onnx version: [coming soon]
***Note 1***: Pre-trained model and sample data cannot be shared because of some personal reasons. You can train the model by yourself with generated dataset (below)

### Data: Generate images and masks
Code for generating is [here](https://github.com/Mushroomcat9998/Cropping-data-generation)
(This repo has not had README.md yet, readers should research it temporarily until instructions are completed)

***Note 2***: This method prioritizes accuracy over speed, so it cannot be used for real-time tasks. 
It takes 0.9s on average to run each sample on CPU Intel® Core™ i7-4770 CPU @ 3.40GHz × 8 (including pre-processing, prediction by model and post-processing)

### How to run
```
python infer.py --data-path "path/to/images/folder" --model-path "path/to/model"
```
