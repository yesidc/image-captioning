#!/usr/bin/env python

from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast, AutoFeatureExtractor, \
    TrainingArguments, Trainer
# todo look into what this data collector does
#from transformers import default_data_collator
import datasets
from PIL import Image
import torch
import torchvision.transforms as transforms
import ipywidgets as widgets
import numpy as np
import evaluate
from IPython.core.display_functions import display

# ## Model


model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    'microsoft/swin-tiny-patch4-window7-224',
    'distilgpt2'
)

print(
    f'This model uses a pre-trained encoder of type {type(model.encoder)} and pre-trained decoder of type {type(model.decoder)}')

#Device

device = torch.device("mps")
model.to(device)

#Database
# The image captioning algorithm is trained on the COCO dataset which consists of images and captions.

COCO_DIR = '/Users/yesidcano/repos/image-captioning/data/coco'
ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=COCO_DIR, split="train[10:70]")


#Data processing
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
# gpt-2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token


# Model config

model.config.pad_token = tokenizer.pad_token
model.config.pad_token_id = tokenizer.pad_token_id

model.config.decoder_start_token = tokenizer.bos_token
model.config.decoder_start_token_id = tokenizer.bos_token_id



# Define the transforms to be applied to the images
transform = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])


def preprocess_fn(examples):
    # Swin expects pixel_values instead of input_ids
    examples['pixel_values'] = [transform(Image.open(path).convert('RGB')) for path in examples['image_path']]
    # todo set this parameter to the average length of the captions
    tokenized = tokenizer(
        examples['caption'], padding='max_length', max_length=50, truncation=True
    )['input_ids']

    # the output captions
    examples['labels'] = [[l if l != tokenizer.pad_token_id else -100 for l in t] for t in tokenized]

    # delete unused keys
    del examples['image_path']
    del examples['caption']
    return examples


processed_dataset = ds.map(
    function=preprocess_fn,
    batched=True,
    batch_size=50,
    # remove_columns=['image_id', 'caption_id', 'caption', 'height', 'width', 'file_name', 'coco_url', 'image_path']

)


# By default data are shuffled.
processed_dataset = processed_dataset.train_test_split(test_size=0.1)

# freeze layer of the encoder (Assuming that the model is already good at understanding images). Since all cross-attention weights need to be optimized
# I do not freeze any of the decoder layers.
for name, param in model.encoder.named_parameters():
    # freeze stage 1 and 2 of the Swin encoder.
    if 'encoder.layer.3' in name:
        break
    param.requires_grad = False





rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")



def metrics(pred_labels):
    """
     Process the predicted captions and the labels (image captions) used to compute the metrics.
    :param pred_labels: set of predicted and target tokens. Type: transformers.trainer_utils.EvalPrediction
    :return: a dictionary with the resulting metric values.
    """
    results={}
    predictions = pred_labels.predictions[0]
    labels = pred_labels.label_ids

    #Convert the model's output into token IDs
    predictions = predictions.argmax(axis=-1)
    # predicted captions are decoded into strings using GPT-2 tokenizer
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # the token ID -100 indicates the end of the sequence.
    # Replaces all -100 values with the id of the padding token in the tokeniezer
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge = rouge_metric.compute(predictions= decoded_predictions,
                                         references= decoded_labels,
                                         use_stemmer=True) # returns a dict

    bleu = bleu_metric.compute(predictions=decoded_predictions,references=[[r] for r in decoded_labels])

    #Round values
    rouge = {k: round(v, 4) for k, v in rouge.items()}
    bleu = {k: [round(p, 4) for p in v ] if isinstance(v, list) else round(v, 4)  for k, v in bleu.items()}
    results.update(rouge)
    results.update((bleu))
    return results


#Training Procedure:



training_arg = TrainingArguments(
    output_dir='../models/swin_image_captioning',  # dicts output
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=64,  # training batch size
    per_device_eval_batch_size=64,  # evaluation batch size
    load_best_model_at_end=True,
    log_level='info',
    logging_steps=50,
    evaluation_strategy='epoch',
    save_strategy='epoch',
)

trainer = Trainer(
    model=model,
    args=training_arg,
    compute_metrics=metrics,
    train_dataset=processed_dataset['train'],
    eval_dataset=processed_dataset['test'],
    #data_collator=default_data_collator
)




trainer.evaluate()




# 4 epochs took 8 hours
trainer.train()


#TODO create graphic of metrics per epoch. x axis epoch number and y axis value of the matric


trainer.save_model()

# need to save the tokenizer
tokenizer.save_pretrained('../models/swin_image_captioning')


# loading model and config from pretrained folder
finetuned_model = VisionEncoderDecoderModel.from_pretrained('../models/swin_image_captioning')

# In[ ]:




inference_transforms = transforms.Compose(
    [

        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)


# a helper function to caption images from the web or a file path
def generate_caption(m, path):
    img = Image.open(path).convert('RGB')
    img_transformed = inference_transforms(img).unsqueeze(0)
    # tensor dimensions max_lenght X num_return_sequences, where ij == some_token_id
    model_output = m.generate(
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
    captions = [tokenizer.decode(g, skip_special_tokens=True).strip() for g in model_output]
    # Show image
    display(img)
    return captions, model_output, img_transformed


captions, model_output, img_transformed = generate_caption(  # Out of sample photo
    finetuned_model, '../data/test_data/000000421195_test.jpg'
)
print()



