#!/usr/bin/env python
# PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=$PYTHONPATH:./model.py python fine_tune_model_swin_gpt-2.py
# HF_HOME='D:\huggingface-cache' PYTHONPATH=$PYTHONPATH:./model.py python fine_tune_model_swin_gpt-2.py
# todo modify environmental variable PYTHONPATH

import logging
from logger_image_captioning import logger
import  os
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import datasets
from datasets import DatasetDict
try:
    from image_captioning_model.model import ImageCaptioningModel
except ModuleNotFoundError:
    from model import ImageCaptioningModel

# Create logger
logger = logging.getLogger('image_captioning')

# set HF_HOME=D:\huggingface-cache  then run the script on a separate command python myscript

def train_model(COCO_DIR, dummy_data=False, device_type='mps'):
    """
    Trains an image captioning model
    Args:
        dummy_data: Whether to train using the full COCO dataset or to test the pipeline using a small sample of the data.
        device: Str. mps or cuda
    """

    # for filename in os.listdir(COCO_DIR):
    #     f = os.path.join(COCO_DIR, filename)
    #     # checking if it is a file
    #     if os.path.isfile(f):
    #         print(f)
    # todo do not load the train set this is slowing down everything and since it is not used this makes no sense to have.
    # Database
    if dummy_data:
        ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir="./dummy_data/")
    else:
        ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=COCO_DIR, split=['train','validation'])
        # crate DatasetDict with the train and validation split
        ds = DatasetDict({'train':ds[0], 'validation':ds[1]})

    logger.info(f'Dataset loaded successfully: {ds}')

    # Create instance of the image captioning model
    image_captioning_model = ImageCaptioningModel()

    image_captioning_model(ds, device_type)
    del ds
    logger.info(image_captioning_model)

    # Training Procedure:
    # reports evaluation metrics at the end of each epoch on top of the training loss.

    training_arg = TrainingArguments(
        output_dir='../models/swin_image_captioning',  # dicts output
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,  # training batch size todo try batch size 4
        per_device_eval_batch_size=4,  # evaluation batch size
        load_best_model_at_end=True,
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
        log_level='info',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        # use_mps_device=True,  # use Apple Silicon
    )

    # Check that the model is on the GPU
    logger.info(
        f'Model is on the GPU {next(image_captioning_model.model.parameters()).device}')  # should print "cuda:0 / mps:0" if on GPU

    trainer = Trainer(
        model=image_captioning_model.model,
        args=training_arg,
        compute_metrics=image_captioning_model.metrics,
        train_dataset=image_captioning_model.processed_dataset['train'],
        eval_dataset=image_captioning_model.processed_dataset['validation'],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # stop training if validation loss stops improving

    # data_collator=default_data_collator
    )

    trainer.evaluate()

    # Resume fine-tuning from the last checkpoint
    # trainer.train(resume_from_checkpoint=True)

    # 4 epochs took 8 hours
    trainer.train()

    logger.info('Finished fine tuning the model')
    trainer.save_model()

    # Save the tokenizer: saves these files preprocessor_config.json, vocab.json special_tokens.json merges.txt
    image_captioning_model.tokenizer.save_pretrained('../models/swin_image_captioning')


# todo change device to cuda
train_model(COCO_DIR='C:\\Users\\yesid\\Documents\\repos\\image-captioning\\data\\coco', dummy_data=False, device_type='cuda')
