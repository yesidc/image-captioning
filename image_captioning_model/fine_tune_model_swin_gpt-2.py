#!/usr/bin/env python
# PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=$PYTHONPATH:./model.py python fine_tune_model_swin_gpt-2.py
# todo modify environmental variable PYTHONPATH

import logging
from logger_image_captioning import logger

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, VisionEncoderDecoderModel
import datasets

try:
    from image_captioning_model.model import ImageCaptioningModel,GenerateCaptions
except ModuleNotFoundError:
    from model import ImageCaptioningModel,GenerateCaptions

# Create logger
logger = logging.getLogger('image_captioning')


def train_model(COCO_DIR, dummy_data=False, device_type='mps'):
    """
    Trains an image captioning model
    Args:
        dummy_data: Whether to train using the full COCO dataset or to test the pipeline using a small sample of the data.
        device: Str. mps or cuda
    """

    # Database
    if dummy_data:
        ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir="./dummy_data/")
    else:
        ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=COCO_DIR, split="validation")

    logger.info(f'Dataset loaded successfully: {ds}')


    def compute_metric():
        evaluation_metrics = GenerateCaptions(VisionEncoderDecoderModel.from_pretrained('../models/swin_image_captioning'))
        evaluation_metrics.evaluate_predictions(ds)
        print()
    compute_metric()
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
        num_train_epochs=1,
        per_device_train_batch_size=16,  # training batch size todo try batch size 4
        per_device_eval_batch_size=16,  # evaluation batch size # todo use bigger batch size
        load_best_model_at_end=True,
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
        log_level='debug',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        # use_mps_device=True,  # use Apple Silicon
    )
    #to compute the number of total steps devide the number of datapoints by the batch size and multiply by the number of epochs

    # Add a file handler to the logger
    file_handler = logging.FileHandler(training_arg.logging_dir + '/training.log')
    logger.addHandler(file_handler)
    # Check that the model is on the GPU
    logger.info(
        f'Model is on the GPU {next(image_captioning_model.model.parameters()).device}')  # should print "cuda:0 / mps:0" if on GPU

    trainer = Trainer(
        model=image_captioning_model.model,
        args=training_arg,
        compute_metrics=image_captioning_model.metrics,
        train_dataset=image_captioning_model.processed_dataset['train'],
        eval_dataset=image_captioning_model.processed_dataset['validation'],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]  # stop training if validation loss stops improving
        # data_collator=default_data_collator
    )

    trainer.evaluate()

    # Resume fine-tuning from the last checkpoint. if by the time of the interruption it was on epoch 5, it continuos from there until num total epochs
    #trainer.train(resume_from_checkpoint=True)

    # 4 epochs took 8 hours
    trainer.train()

    logger.info('Finished fine tuning the model')
    trainer.save_model()

    # Save the tokenizer: saves these files preprocessor_config.json, vocab.json special_tokens.json merges.txt
    image_captioning_model.tokenizer.save_pretrained('../models/swin_image_captioning')


# todo change device to cuda
train_model(COCO_DIR='/Users/yesidcano/repos/image-captioning/data/coco', dummy_data=True)
