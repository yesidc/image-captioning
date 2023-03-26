import os
import glob
import time
import logging
import datasets
from transformers import TrainerCallback, VisionEncoderDecoderModel

from image_captioning_model.model import GenerateCaptions


# Create logger
logger = logging.getLogger('image_captioning')

def load_dataset(COCO_DIR, dummy_data=False):

    # Database
    if dummy_data:
        ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir="./dummy_data/")
    else:
        ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=COCO_DIR)
    return ds




class CustomCallbackStrategy(TrainerCallback):
    def __init__(self,output_dir,validation, trainer, *args,**kwargs ):
        self.output_dir = output_dir
        self.validation = validation
        self.trainer= trainer



    def on_epoch_end(self, args, state, control, **kwargs):
        # Call the on_epoch_end method of the parent class
        super().on_epoch_end(args, state, control, **kwargs)
        control.should_save = True  # ensure that the model is saved at the end of the epoch
        # self.trainer = state.trainer
        self.trainer.save_model(args.output_dir)  # save the model checkpoint using the Trainer instance
        # load the model checkpoint to compute the evaluation metrics
        evaluation_metrics = GenerateCaptions(
            VisionEncoderDecoderModel.from_pretrained(self.output_dir))

        start_time = time.time()
        # compute the evaluation metrics on the validation set
        evaluation_metrics.evaluate_predictions(self.validation, args.eval_batch_size)
        end_time = time.time()
        logger.debug(f'Time taken to compute metrics: {end_time - start_time}')
        logger.info(f"Metric's results: {evaluation_metrics.results}")
        logger.info(f"End of epoch {state.epoch}.")






