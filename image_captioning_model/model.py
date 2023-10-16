import os
from logger_image_captioning import logger
import logging
from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import evaluate
from functools import lru_cache
from joblib import Memory
import time
from tqdm import tqdm


# to cache on disk the validation data
mem = Memory(location='../cache_validation', verbose=0)
# create logger
logger = logging.getLogger('image_captioning')


class ComputeMetricMixin():
    def compute_metric(self):
        """
        Computes rouge and bleu metrics
        :param decoded_predictions: Python list of decoded predictions
        :param decoded_labels: Python list of decoder labels

        """
        self.results = {}
        rouge_metric = evaluate.load("rouge")
        bleu_metric = evaluate.load("bleu")
        rouge = rouge_metric.compute(predictions=self.decoded_predictions,
                                     references=self.decoded_labels,
                                     use_stemmer=True)  # returns a dict

        bleu = bleu_metric.compute(predictions=self.decoded_predictions,
                                   references=self.decoded_labels)

        # Round values
        rouge = {k: round(v, 4) for k, v in rouge.items()}
        bleu = {k: [round(p, 4) for p in v] if isinstance(v, list) else round(v, 4) for k, v in bleu.items()}
        self.results.update(rouge)
        self.results.update((bleu))
        self.decoded_predictions = None
        self.decoded_labels = None


class MetricsMixin(ComputeMetricMixin):

    def metrics(self, pred_labels):
        """
         Process the predicted captions and the labels (image captions) used to compute the metrics.
        :param pred_labels: EvaluationPredition object: named tuple with predictions and label_ids field.
        :return: a dictionary with the resulting metric values.
        """
        predictions = pred_labels.predictions[0]
        labels = pred_labels.label_ids

        # Convert the model's output into token IDs
        predictions = predictions.argmax(axis=-1)
        # predicted captions are decoded into strings using GPT-2 tokenizer
        self.decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # the token ID -100 indicates the end of the sequence.
        # Replaces all -100 values with the id of the padding token in the tokenizer
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        self.decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        self.compute_metric()
        return self.results

# Define a mixin class to preprocess the data
class DataSetMixin():
    def processed_dataset(self, ds):
        logger.info('Start data preprocessing (mapping operation)')
        # apply the preprocessing function to the dataset
        processed_dataset = ds.map(
            function=self.preprocess_fn,
            batched=True,
            remove_columns=ds['train'].column_names,
            # cache_file_names="processed_cache_10000_val.arrow" #name of the file where the processed data is to be cached
            # batch_size=50,
            # remove_columns=['image_id', 'caption_id', 'caption', 'height', 'width', 'file_name', 'coco_url', 'image_path'],

        )
        logger.info(f'Data processing finished {processed_dataset}')
        # processed_dataset = processed_dataset.train_test_split(test_size=0.05)
        return processed_dataset


class DataProcessing():
    """
    A class that encapsulates the data preprocessing
    """
    def __init__(self):
        # gpt-2 tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.counter = 0
        # Define the transforms to be applied to the images
        self.transform = transforms.Compose([

            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])

    def preprocess_fn(self, examples):

        # Swin expects pixel_values instead of input_ids
        examples['pixel_values'] = [self.transform(Image.open(path).convert('RGB')) for path in examples['image_path']]
        # print(examples[0]['image_path'])
        # todo set this parameter to the average length of the captions
        tokenized = self.tokenizer(
            examples['caption'], padding='max_length', max_length=50, truncation=True
        )['input_ids']

        # the output captions
        examples['labels'] = [[l if l != self.tokenizer.pad_token_id else -100 for l in t] for t in tokenized]

        # delete unused keys
        del examples['image_path']
        del examples['caption']
        return examples


class ImageCaptioningModel(DataSetMixin, MetricsMixin, DataProcessing):
    """
    A class that encapsulates the entire image captioning model
    """
    # Model
    model= None

    # Model config
    def model_config(self):

        ImageCaptioningModel.model.config.pad_token = self.tokenizer.pad_token
        ImageCaptioningModel.model.config.pad_token_id = self.tokenizer.pad_token_id

        ImageCaptioningModel.model.config.decoder_start_token = self.tokenizer.bos_token
        ImageCaptioningModel.model.config.decoder_start_token_id = self.tokenizer.bos_token_id

    # freeze layer of the encoder (Assuming that the model is already good at understanding images). Since all cross-attention weights need to be optimized
    # I do not freeze any of the decoder layers.
    def freeze_layer(self):
        for name, param in ImageCaptioningModel.model.encoder.named_parameters():
            # freeze stage 1  the Swin encoder.
            if 'encoder.layer.3' in name:
                break
            param.requires_grad = False

    def __call__(self, ds, device_type,start_from_checkpoint=False,
                 path_to_checkpoint=None):

        if start_from_checkpoint:
            ImageCaptioningModel.model = VisionEncoderDecoderModel.from_pretrained(path_to_checkpoint)
            logger.info('Model loaded from checkpoint')
        else:
            ImageCaptioningModel.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                'microsoft/swin-tiny-patch4-window7-224',
                'distilgpt2'
            )


        if device_type == 'mps':
            device = torch.device("mps")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f'Device {device}')

        ImageCaptioningModel.model.to(device)

        self.model_config()
        self.freeze_layer()
        self.processed_dataset = self.processed_dataset(ds)

    def __str__(self):

        return f'This model uses a pre-trained encoder of type {type(ImageCaptioningModel.model.encoder)} and pre-trained decoder of type {type(ImageCaptioningModel.model.decoder)}'


class GenerateEvaluateCaptions(ComputeMetricMixin, DataProcessing):

    def __init__(self, tuned_model):

        self.tuned_model = tuned_model
        super().__init__()

    def read_img_predict(self, path, evaluate):
        # Only cache if evaluate is True
        images_to_predict = None

        @mem.cache
        def process_img(path):
            img = Image.open(path)
            if img.mode != "RGB":
                img = Image.open(path).convert('RGB')
            return self.transform(img).unsqueeze(0)

        # tensor dimensions max_lenght X num_return_sequences, where ij == some_token_id
        # todo use kwargs to pass these parameters
        if evaluate:
            batch = []
            for p in path:
                batch.append(process_img(p).squeeze(0))  # remove the batch dimension
            images_to_predict = torch.stack(batch)  # add the batch dimension
            batch = None

       # generate captions
        model_output = self.tuned_model.generate(
            images_to_predict if evaluate else process_img.func(path), # if evaluate is True, then use the cached images
            # num_beams=3,
            # max_length=15,
            # early_stopping=True,
            # do_sample=True,
            # top_k=10,
            num_return_sequences=1,
        )
        images_to_predict = None
        torch.cuda.empty_cache()  # free up memory
        # g is a tensor like this one: tensor([50256,    13,   198,   198,   198,   198,   198,   198,   198, 50256,
        # 50256, 50256, 50256, 50256, 50256])
        captions = [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in model_output]
        return captions

    # a helper function to generate captions

    def generate_caption(self, path):
        """Generate captions for a single image or a directory of images
        :param path: path to a single image or a directory of images
        :return: a list of captions
        """

        if os.path.isdir(path):
            self.decoded_predictions = []
            for root, dirs, files in os.walk(path):
                for file in tqdm(files):
                    self.decoded_predictions.append(self.read_img_predict(os.path.join(root, file), False))

            return self.decoded_predictions

        elif os.path.isfile(path):
            return self.read_img_predict(path, False)
        else:
            pass

    @lru_cache(maxsize=None)
    def group_data_per_id(self, validation_split):
        """"
        Group the data per image id. This is used to evaluate the model.
        :param validation_split: the validation split
        :return: a dictionary with the image id as key and a list of indexes where the same image occurs as value
        """
        groups = {}
        # group images per img_id
        for i in range(validation_split.shape[0]):
            img_id = validation_split[i]['image_id']
            if img_id not in groups:
                groups[img_id] = {
                    'index': []  # list of indexes where the same image occurs
                }
            groups[img_id]['index'].append(i)
        return groups

    def evaluate_predictions(self, validation_split, batch_size=4):
        """

        :param validation_split: data to evaluate the model
        :param batch_size: Number of images to evaluate at once

        """
        # list of references. Contains nested lists with all captions that belong to a single image
        logging.getLogger("transformers").setLevel(logging.ERROR)
        img_paths = []
        count = 0
        self.decoded_predictions = []
        self.decoded_labels = []

        start_time = time.time()
        groups = self.group_data_per_id(validation_split)
        end_time = time.time()
        logger.debug(f'Time taken to group the validation data: {end_time - start_time}')

        for k in tqdm(groups.keys()):

            img_paths.append(validation_split[groups[k]['index'][0]]['image_path'])
            count += 1
            reference_captions = []
            # iterate across all the datapoints with the same img_id and extract all captions for a single image
            for c in groups[k]['index']:
                reference_captions.append(validation_split[c]['caption'])
            self.decoded_labels.append(reference_captions)

            if count == batch_size or k == list(groups.keys())[-1]:
                captions = self.read_img_predict(img_paths, evaluate=True)
                _ = [self.decoded_predictions.append(c) for c in captions]
                del _
                img_paths = []

                count = 0

        self.compute_metric()
        logging.getLogger("transformers").setLevel(logging.DEBUG)
