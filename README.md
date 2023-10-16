# image-captioning
Under construction.


How to use this code:

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

```
## Dataset
The image captioning model can be trained on the following datasets:

- [COCO](https://cocodataset.org/#home)
- [Flickr8k](https://www.kaggle.com/adityajn105/flickr8k)
- [Flickr30k](https://www.kaggle.com/hsankesara/flickr-image-dataset)

### Dummy data
For testing purposes you can use `dummy_data`. This is a small dataset with 80 training and validation samples and 16 test samples. The dataset is available on huggingface:
```
https://huggingface.co/datasets/ydshieh/coco_dataset_scrip
```



