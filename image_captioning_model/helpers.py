import os
import csv
import time
import logging
import datasets
from datasets import Dataset, DatasetDict
from transformers import TrainerCallback, VisionEncoderDecoderModel
import pandas as pd
from image_captioning_model.model import GenerateCaptions

# Create logger
logger = logging.getLogger('image_captioning')


def load_create_ds_flickr_8k(PATH_DATASET):
    """
    Load Flickr 8k dataset
    :param PATH_DATASET: Path to the dataset. The directory should contain the following files: `Flickr8k.token.txt` and `images` directory.
    :return: A dataset object
    """
    # Load Flickr 8k dataset
    # Load captions from file

    data = []
    MIN_CAPTION = 20
    # IMAGES_PATH = '../data/flicker_8k/images'
    # caption_file = "../data/flicker_8k/Flickr8k.token.txt"
    if not os.path.isdir(os.path.join(PATH_DATASET, "images")):
        raise FileNotFoundError("Directory 'images' not found in directory. Save all images to a directory called "
                                "`images`.")

    if not os.path.isfile(os.path.join(PATH_DATASET, "Flickr8k.token.txt")):
        raise FileNotFoundError("File 'Flickr8k.token.txt' not found in directory")
    caption_file = PATH_DATASET + "/Flickr8k.token.txt"
    path_images = PATH_DATASET + "/images"
    with open(caption_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            image_name, caption = row
            # Each image name has a suffix `#(caption_number)`
            image_id = os.path.splitext(image_name)[0]
            image_name = image_name.split("#")[0]
            image_path = os.path.join(path_images, image_name)
            if  not os.path.isfile(os.path.join(image_path)):
                print(f'The file "{image_name}" does not exists in directory "{image_name}"')
                continue




            if image_name.endswith("jpg"):

                caption = caption.replace(' .', '').strip()
                tokens = caption.strip().split()
                if len(tokens) < MIN_CAPTION :
                    #print(caption)
                    continue

            data.append({'image_id': image_id, 'image_path': image_path, 'caption': caption})

    ds = datasets.Dataset.from_pandas(pd.DataFrame(data=data))
    # Split the dataset into train and validation
    train_testvalid = ds.train_test_split(test_size=0.1)
    ds =  DatasetDict({'train':train_testvalid['train'], 'validation':train_testvalid['test'] })


    return ds

#'/Users/yesidcano/repos/image-captioning/data/flicker_8k/images/2258277193_586949ec62.jpg.1'
def load_crate_dsflickr_30k(PATH_DATASET):
    # Load captions from file
    import csv
    data = []
    MIN_CAPTION = 5
    IMAGES_PATH = '../data/flicker_30k/images'
    caption_file = "/Users/yesidcano/Downloads/results.csv"
    drop_first = True
    with open(caption_file, newline='') as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            row_data = row[0].split('|')
            try:

                image_name, caption = row_data[0], row_data[2]
                # Each image name has a suffix `#(caption_number)`
            except IndexError:
                print(f'Corrupted datapoint-------------{row_data}--------------')
                continue

            image_id = os.path.splitext(image_name)[0]
            image_path = os.path.join(IMAGES_PATH, image_name.strip())

            caption = caption.replace(' .', '').strip()
            caption = caption.strip()
            tokens = caption.strip().split()
            if len(tokens) < MIN_CAPTION or drop_first:
                drop_first = False
                # print(caption)
                continue

            data.append({'image_id': image_id, 'image_path': image_path, 'caption': caption})

    ds = datasets.Dataset.from_pandas(pd.DataFrame(data=data))
    return ds


def load_dataset(PATH_DATASET, dataset_type=None,dummy_data=False):
    # Database
    if dummy_data:
        ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir="./dummy_data/")
    elif dataset_type == 'coco':
        ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=PATH_DATASET)
    elif dataset_type == 'flickr_8k':
        ds = load_create_ds_flickr_8k(PATH_DATASET)
    elif dataset_type == 'flickr_30k':
        ds = load_crate_dsflickr_30k(PATH_DATASET)
    else:
        raise ValueError("No dataset selected. Please select a dataset to load: 'coco', 'flickr_8k', 'flickr_30k'  ")

    return ds


class CustomCallbackStrategy(TrainerCallback):
    def __init__(self, output_dir, validation, trainer, *args, **kwargs):
        self.output_dir = output_dir
        self.validation = validation
        self.trainer = trainer

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
