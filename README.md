# Image Captioning Model
(Under construction)

This repository contains the code for training an End-to-end Transformer based image captioning model, where both the encoder and decoder use standard pre-trained transformer architectures.

The model can be downloaded from huggingface:
```https://huggingface.co/yesidcanoc/image-captioning-swin-tiny-distilgpt2```

## Model Architecture
### Encoder
The encoder uses the pre-trained Swin transformer (Liu et al., 2021) that is a general-purpose backbone for computer vision. It outperforms ViT, DeiT and ResNe(X)t models at tasks such as image classification, object detection and semantic segmentation. The fact that this model is not pre-trained to be a 'narrow expert'--- a model pre-trained to perform a specific task e.g., image classification --- makes it a good candidate for fine-tuning on a downstream task

### Decoder

Distilgpt2



1. Clone the repository
2. Create Virtual Environment and install the requirements using: `pip install -r requirements.txt`
3. Download the dataset:
    - [COCO](https://cocodataset.org/#home)
    - [Flickr8k](https://www.kaggle.com/adityajn105/flickr8k)
    - [Flickr30k](https://www.kaggle.com/hsankesara/flickr-image-dataset)
4. Train the model

## Dataset
The image captioning model can be trained and subsequently evaluated on the following datasets:

- [COCO](https://cocodataset.org/#home)
- [Flickr8k](https://www.kaggle.com/adityajn105/flickr8k)
- [Flickr30k](https://www.kaggle.com/hsankesara/flickr-image-dataset)

### Dummy data
For testing purposes you can use `dummy_data`. This is a small dataset with 80 training and validation samples and 16 test samples. The dataset is available on huggingface:
```
https://huggingface.co/datasets/ydshieh/coco_dataset_scrip
```

## Training

Execute the following command to train the model:
```
python train.py 
```
`train.py` take the following arguments:
```     
--path_dataset: path to the dataset
--dummy_data: use dummy data for testing purposes; default is False.
--path_to_checkpoint: path to the checkpoint. Default is None.
--start_from_checkpoint: Boolean, start from checkpoint. Default is False.
--resume_from_checkpoint: Boolean, resume from checkpoint. Default is False.
--output_dir: path to the output directory. default='../models/swin_gpt-2-image-captioning'
--dataset_type: type of dataset. Default is 'coco'. Other option is 'flickr_8k', flickr_30k
--num_train_epochs: number of training epochs. Default is 3.

```

### Trainer

This work uses the `Transformers` `Trainer` class designed to fine-tune pretrained transformer based models.
  

To conduct the fine-tuning phase, the `Trainer` class was instantiated with a pretrained model, metrics, training and validation data, and a set of training arguments.
The pretrained model is an instance of the `VisionEncoderDecoderModel` class. This class, `VisionEncoderDecoderModel`, allows initializing an image-to-text-sequence model where the encoder is a pretrained transformer-based vision autoencoding model and the decoder is a backbone of a pretrained text autoregressive model. Note that `VisionEncoderDecoderModel` class randomly initializes the cross-attention weights, hence these must be fine-tuned on a downstream task. 

The training and evaluation split used  to generate an instance of the `Trainer` class consist of two DataSet objects with two features: pixel values and labels. 

Lastly, to initialize the `Trainer` an instance of the `TrainingArguments` class is provided.The hyperparameters needed to fine-tune the model, which are subsequently consumed by the `Trainer` class, are set via an instance of the `TrainingArguments` class. For this project, most of the hyperparameters (e.g., learning rate, loss function) used for fine-tuning are provided by the training API.


## Metrics
The model is evaluated on the following metrics:
- BLEU
- ROUGE

## Logs
The logs are saved in the `image_captioning_model/image_captioning.log` file.
## References
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ArXiv. /abs/2103.14030