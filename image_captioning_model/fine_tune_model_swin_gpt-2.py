#!/usr/bin/env python
# PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=$PYTHONPATH:./model.py python fine_tune_model_swin_gpt-2.py
# HF_HOME='D:\huggingface-cache' PYTHONPATH=$PYTHONPATH:./model.py python fine_tune_model_swin_gpt-2.py


import logging
from helpers import CustomCallbackStrategy
from logger_image_captioning import logger
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, VisionEncoderDecoderModel
from helpers import load_dataset
try:
    from image_captioning_model.model import ImageCaptioningModel, GenerateCaptions
except ModuleNotFoundError:
    from model import ImageCaptioningModel, GenerateCaptions

# Create logger
logger = logging.getLogger('image_captioning')


# set HF_HOME=D:\huggingface-cache  then run the script on a separate command python myscript

def train_model(COCO_DIR, output_dir, dummy_data=False, device_type='mps'):
    """
    Trains an image captioning model
    :param COCO_DIR: path to the COCO dataset
    :param output_dir: path to the output directory
    :param dummy_data: if True, uses a dummy dataset
    :param device_type: device type to use for training
    """

    ds = load_dataset(COCO_DIR, dummy_data)

    logger.info(f'Dataset loaded successfully: {ds}')

    # Create instance of the image captioning model
    image_captioning_model = ImageCaptioningModel()

    image_captioning_model(ds, device_type)
    logger.info(image_captioning_model)

    # Training Procedure:
    # reports evaluation metrics at the end of each epoch on top of the training loss.

    training_arg = TrainingArguments(
        output_dir=output_dir,  # dicts output
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=10,  # training batch size
        per_device_eval_batch_size=10,  # evaluation batch size
        load_best_model_at_end=True,
        # warmup_steps=10000,
        # dataloader_num_workers=24,  # this machine has 24 cpu cores
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
        # fp16=True,
        log_level='info',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        # use_mps_device=True,  # use Apple Silicon

    )
    # to compute the number of total steps devide the number of datapoints by the batch size and multiply by the number of epochs

    trainer = Trainer(
        model=image_captioning_model.model,
        args=training_arg,
        # compute_metrics=image_captioning_model.metrics,
        train_dataset=image_captioning_model.processed_dataset['train'],
        eval_dataset=image_captioning_model.processed_dataset['validation'],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        # stop training if validation loss stops improving

        # data_collator=default_data_collator
    )

    logger.info('Starting trainer.evaluate()')
    # trainer.evaluate()

    # Resume fine-tuning from the last checkpoint
    # trainer.train(resume_from_checkpoint=True)
    # Check that the model is on the GPU
    logger.info(
        f'Model is on the GPU {next(image_captioning_model.model.parameters()).device}. STARTING TRAINING')  # should print "cuda:0 / mps:0" if on GPU
    custom_callback = CustomCallbackStrategy(output_dir=output_dir,
                                             validation=ds['validation'],
                                             trainer=trainer)

    trainer.add_callback(custom_callback)
    trainer.train()

    logger.info('Finished fine tuning the model')
    trainer.save_model()

    # Save the tokenizer: saves these files preprocessor_config.json, vocab.json special_tokens.json merges.txt
    image_captioning_model.tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    train_model(COCO_DIR='../data/coco', dummy_data=True,
                device_type='mps', output_dir='../models/swin_NO_F_GPT_image_captioning')

# if __name__ == '__main__':
#     # todo change device to cuda
#     train_model(COCO_DIR='C:\\Users\\yesid\\Documents\\repos\\image-captioning\\data\\coco', dummy_data=False,
#                 device_type='cuda', output_dir='../models/swin_NO_F_GPT_image_captioning')
#
#
