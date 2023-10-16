#!/usr/bin/env python
# PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=$PYTHONPATH:./model.py python train.py --> to use MPS run this command
# HF_HOME='D:\huggingface-cache' PYTHONPATH=$PYTHONPATH:./model.py python train.py


import argparse
from logger_image_captioning import logger
import logging
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from helpers import load_dataset
try:
    from image_captioning_model.model import ImageCaptioningModel
    from image_captioning_model.generate_captions import generate_captions_and_evaluate
except ModuleNotFoundError:
    from model import ImageCaptioningModel
    from generate_captions import generate_captions_and_evaluate


# Create logger
logger = logging.getLogger('image_captioning')




# set HF_HOME=D:\huggingface-cache  then run the script on a separate command python myscript

def train_model(output_dir,
                ds,
                device_type='mps',
                start_from_checkpoint=False,
                path_to_checkpoint=None,
                resume_from_checkpoint=False,):
    """
    Trains an image captioning model
    :param output_dir: path to the output directory
    :param device_type: device type to use for training
    """

    # Create instance of the image captioning model
    image_captioning_model = ImageCaptioningModel()

    image_captioning_model(ds=ds, device_type=device_type,start_from_checkpoint=start_from_checkpoint,path_to_checkpoint=path_to_checkpoint)
    logger.info(image_captioning_model)

    # Training Procedure:
    # reports evaluation metrics at the end of each epoch on top of the training loss.

    training_arg = TrainingArguments(
        output_dir=output_dir,  # dicts output
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=10,  # training batch size
        per_device_eval_batch_size=10,  # evaluation batch size
        load_best_model_at_end=True,
        # optim='adamw_torch',

        # learning_rate=5e-6,
        # warmup_steps=10000,
        # dataloader_num_workers=24,  # this machine has 24 cpu cores
        logging_dir='./logs',  # directory for storing logs
        logging_steps=1,
        # fp16=True,
        log_level='info',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        use_mps_device=True,  # use Apple Silicon

    )
    # to compute the number of total steps devide the number of datapoints by the batch size and multiply by the number of epochs

    # define optimizer
    # optimizer = torch.optim.AdamW(image_captioning_model.model.parameters(),
    #                               lr=training_arg.learning_rate,
    #                               betas=(0.9, 0.98),)
    # scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=training_arg.warmup_steps)
    # Create trainer
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
    # trainer.lr_scheduler = scheduler
    # trainer.optimizer = optimizer
    logger.info('Starting trainer.evaluate()')
    trainer.evaluate()


    # Check that the model is on the GPU
    logger.info(
        f'Model is on the GPU {next(image_captioning_model.model.parameters()).device}. STARTING TRAINING')  # should print "cuda:0 / mps:0" if on GPU

    # compute metrics on the validation set
    # custom_callback = CustomCallbackStrategy(output_dir=output_dir,
    #                                          validation=ds['validation'],
    #                                          trainer=trainer)
    # trainer.add_callback(custom_callback)

    if resume_from_checkpoint:
        logger.info('Resuming training from checkpoint')
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    logger.info('Finished fine tuning the model')


    trainer.save_model()

    # Save the tokenizer: saves these files preprocessor_config.json, vocab.json special_tokens.json merges.txt
    image_captioning_model.tokenizer.save_pretrained(output_dir)

    del image_captioning_model



if __name__ == '__main__':
    # Create parser
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--path_dataset', type=str, default=None, help='Path to the dataset')
    parser.add_argument('--dummy_data', type=bool, default=False, help='Use dummy data')
    parser.add_argument('--path_to_checkpoint', type=str, default=None, help='Path to the checkpoint')
    parser.add_argument('--start_from_checkpoint', type=bool, default=False, help='Start from checkpoint')
    parser.add_argument('--resume_from_checkpoint', type=bool, default=False, help='Resume from checkpoint')
    parser.add_argument('--output_dir', type=str, default='../models/swin_gpt-2-image-captioning',
                        help='Output directory')
    parser.add_argument('--dataset_type', type=str, default='coco',
                        help='Type of dataset to use: flickr_8k, flickr_30k, coco')

    args = parser.parse_args()


    # num_epochs = 3
    PATH_DATASET = args.path_dataset
    dummy_data = args.dummy_data
    path_to_checkpoint = args.path_to_checkpoint
    start_from_checkpoint = args.start_from_checkpoint
    dataset_type = args.dataset_type
    output_dir = args.output_dir
    cache_checkpoint_dir = '../models/checkpoints' # directory to store checkpoints
    ds = load_dataset(PATH_DATASET=PATH_DATASET, dummy_data=dummy_data, dataset_type=dataset_type)
    logger.info(f'Dataset {dataset_type} loaded successfully: {ds}')
    # validation_data = ds['validation']
    train_model(device_type='mps',
                ds=ds,
                output_dir=output_dir,
                start_from_checkpoint=start_from_checkpoint,
                path_to_checkpoint=path_to_checkpoint,
                resume_from_checkpoint=False)

