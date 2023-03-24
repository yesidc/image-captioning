from logger_image_captioning import logger
import logging
from transformers import  VisionEncoderDecoderModel, GPT2TokenizerFast
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import evaluate
from IPython.core.display_functions import display



# create logger
logger = logging.getLogger('image_captioning')

class MetricsMixin():

    def metrics(self, pred_labels):
        """
         Process the predicted captions and the labels (image captions) used to compute the metrics.
        :param pred_labels: EvaluationPredition object: named tupple with predictions and label_ids field.
        :return: a dictionary with the resulting metric values.
        """
        print(f'Number of datapoints passed to the metrics function{pred_labels.label_ids.shape}')
        print(f'Predictions shape: {pred_labels.predictions[0].shape}')

        rouge_metric = evaluate.load("rouge")
        bleu_metric = evaluate.load("bleu")
        results = {}
        predictions = pred_labels.predictions[0]
        labels = pred_labels.label_ids

        # Convert the model's output into token IDs
        predictions = predictions.argmax(axis=-1)
        # predicted captions are decoded into strings using GPT-2 tokenizer
        decoded_predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        del predictions
        # the token ID -100 indicates the end of the sequence.
        # Replaces all -100 values with the id of the padding token in the tokeniezer
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        del labels
        rouge = rouge_metric.compute(predictions=decoded_predictions,
                                     references=decoded_labels,
                                     use_stemmer=True)  # returns a dict

        bleu = bleu_metric.compute(predictions=decoded_predictions, references=[[r] for r in decoded_labels])

        # Round values
        rouge = {k: round(v, 4) for k, v in rouge.items()}
        bleu = {k: [round(p, 4) for p in v] if isinstance(v, list) else round(v, 4) for k, v in bleu.items()}
        results.update(rouge)
        results.update((bleu))
        return results

# cached "processed_cache_10000_val.arrow",

class DataSetMixin():
    def processed_dataset(self, ds):
        logger.info('Start data preprocessing (mapping operation)')
        processed_dataset = ds.map(
            function=self.preprocess_fn,
            batched=True,
            remove_columns=ds['train'].column_names,
            #cache_file_names="processed_cache_10000_val.arrow" #name of the file where the processed data is to be cached
            # batch_size=50,
            # remove_columns=['image_id', 'caption_id', 'caption', 'height', 'width', 'file_name', 'coco_url', 'image_path']

        )
        logger.info(f'Data preprocessing finished {processed_dataset}')
        # processed_dataset = processed_dataset.train_test_split(test_size=0.05)
        return processed_dataset


class DataProcessing():
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
    # 'D:\\huggingface-cache\\datasets\\downloads\\extracted\\0597725cffd24e89bfdd4a70cd41da03bd03b1a51103231a562152480846df2b\\train2017\\000000203564.jpg',
    # 'C:\\Users\\yesid\\.cache\\huggingface\\datasets\\downloads\\extracted\\0597725cffd24e89bfdd4a70cd41da03bd03b1a51103231a562152480846df2b\\train2017\\000000203564.jpg'
    #  'C:\\Users\\yesid\\.cache\\huggingface\\datasets\\downloads\\extracted\\0597725cffd24e89bfdd4a70cd41da03bd03b1a51103231a562152480846df2b\\train2017\\000000005247.jpg'
    def preprocess_fn(self, examples):

        if self.counter == 0:
            logger.info(f"CACHE PATH: {examples['image_path'][0]}")
            self.counter +=1


        # Swin expects pixel_values instead of input_ids
        examples['pixel_values'] = [self.transform(Image.open(path).convert('RGB')) for path in examples['image_path']]
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
    # Model
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        'microsoft/swin-tiny-patch4-window7-224',
        'distilgpt2'
    )


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
            if 'encoder.layer.2' in name:
                break
            param.requires_grad = False

    def __call__(self, ds, device_type, *args, **kwargs):

        if device_type == 'mps':
            device = torch.device("mps")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f'Device {device}')

        ImageCaptioningModel.model.to(device)

        self.model_config()
        #self.freeze_layer()
        self.processed_dataset = self.processed_dataset(ds)

    def __str__(self):

        return f'This model uses a pre-trained encoder of type {type(ImageCaptioningModel.model.encoder)} and pre-trained decoder of type {type(ImageCaptioningModel.model.decoder)}'


class GenerateCaptions(DataProcessing):

    # a helper function to generate captions
    def __call__(self, tuned_model, path, *args, **kwargs):
        img = Image.open(path)
        if img.mode != "RGB":
            img = Image.open(path).convert('RGB')
        img_transformed = self.transform(img).unsqueeze(0)
        # tensor dimensions max_lenght X num_return_sequences, where ij == some_token_id
        # todo use kwargs to pass these parameters
        model_output = tuned_model.generate(
            img_transformed,
            num_beams=3,
            max_length=15,
            early_stopping=True,
            do_sample=True,
            top_k=10,
            num_return_sequences=5,
        )
        # g is a tensor like this one: tensor([50256,    13,   198,   198,   198,   198,   198,   198,   198, 50256,
        # 50256, 50256, 50256, 50256, 50256])
        captions = [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in model_output]
        # Show image
        display(img)
        return captions, model_output, img_transformed
