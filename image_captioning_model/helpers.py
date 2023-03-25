import os
import glob
import time
import logging
from transformers import TrainerCallback, VisionEncoderDecoderModel

from image_captioning_model.model import GenerateCaptions


# Create logger
logger = logging.getLogger('image_captioning')

# def get_last_epoch(directory_path):
#
#
#     # Use glob to find all subdirectories within the directory_path
#     all_subdirs = glob.glob(os.path.join(directory_path, "*"))
#
#     # Filter the subdirectories to only include folders
#     all_subdirs = [f for f in all_subdirs if os.path.isdir(f)]
#
#     # Sort the folders by creation time and get the latest folder
#     latest_folder = max(all_subdirs, key=os.path.getctime)
#
#     # Print the path to the latest folder
#     print("Path to last epoch directory: ", latest_folder)
#     return latest_folder

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

        evaluation_metrics = GenerateCaptions(
            VisionEncoderDecoderModel.from_pretrained(self.output_dir))

     # evaluation_metrics = GenerateCaptions(
     #        VisionEncoderDecoderModel.from_pretrained(get_last_epoch(self.output_dir)))
     #



        start_time = time.time()
        evaluation_metrics.evaluate_predictions(self.validation)
        end_time = time.time()
        logger.debug(f'Time taken to compute metrics: {end_time - start_time}')
        logger.info(f"Metric's results: {evaluation_metrics.results}")
        logger.info(f"End of epoch {state.epoch}.")






