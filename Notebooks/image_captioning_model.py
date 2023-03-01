#!/usr/bin/env python


import nltk
from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast, AutoFeatureExtractor, \
    TrainingArguments, Trainer
# todo look into what this data collector does
#from transformers import default_data_collator
import datasets
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import ipywidgets as widgets
import evaluate


# ## Model


model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    'microsoft/swin-tiny-patch4-window7-224',
    'distilgpt2'
)

print(
    f'This model uses a rre-trained encoder of type {type(model.encoder)} and pre-trained decoder of type {type(model.decoder)}')

# ## Device


device = torch.device("mps")

model.to(device)

# ## Database
# The image captioning algorithm is trained on the COCO dataset which consists of images and captions.


# the datasets.load_dataset manages everything related to caching. So I have to use it.
COCO_DIR = '/Users/yesidcano/repos/image-captioning/data/coco'

ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=COCO_DIR, split="train[10:70]")

# ## Data processing


feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
# gpt-2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token

# In[9]:


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
    # We are padding tokens here instead of using a datacollator
    # max_length is the number of tokens a sentence/caption is made of. this results in a prediction tensor batch_size X max_length X 50257
    # todo change the target length, it can be that low, because the captions longer than 10 is chopped. Maybe set to the
    # todo average token size of the captions in the dataset. CHECK IF LOTS OF PADDING IS ADDED IF THE THE MAX_LENGTH IS TOO BIG.
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

# In[13]:


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
# In[18]:


# throws this error --> TypeError: argument 'ids': 'list' object cannot be interpreted as an integer
import numpy as np




# def process_text_metrics(predictions, labels):
#     # preds = [pred.strip() for pred in preds]
#     # labels = [label.strip() for label in labels]
#
#     # strip text and add new line
#     predictions = ["\n".join(nltk.sent_tokenize(p.strip())) for p in predictions]
#     labels = ["\n".join(nltk.sent_tokenize(l.strip())) for l in labels]
#
#     return predictions, labels


def metrics(pred_labels):
    """
     Process the predicted captions and the labels (image captions) used to compute the metrics.
    :param pred_labels: set of predicted and target tokens. Type: transformers.trainer_utils.EvalPrediction
    :return: a dictionary with the resulting metric values.
    """
    results={}

    # labels contains token ids of the ground truth/captions. ndarray dim = number-of-test-data-points X numnber-of-ids- each per token in the caption
    # pred[0] ndarray {6, 10, 50257} in this case there are 6 test data points, hence 6 predictions to compare againts
    #predictions, labels = pred_labels

    # numpy.ndarray
    #predictions = predictions[0]
    predictions = pred_labels.predictions[0]
    labels = pred_labels.label_ids

    """
     takes the matrix 10 x 50257 and per each row vector extracts the index with the highest probability??? or with the highest value???
   
    TODO do softmax befere doing the argmax?? The softmax is just to see how confiedent the model is in its prediction
     since softmax computes a distribution accross all the vocabulary, if I do softmax and then max I get the probability according to which
    that specific token was selected among the 50257
     This proves my point
     preds1 = predictions.argmax(axis=-1)
     preds2 = softmax(predictions, axis=-1)
     arg_preds2 = preds2.argmax(axis=-1)
     preds1==arg_preds2 --> this is true, hence either I do softmax or not is irrelevant for the extraction of the ids for the decoding.
     
    """
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
    # todo this makes no sense, the length is always the same

    results.update(rouge)
    results.update((bleu))
    return results


# ## Training Procedure:
# 
# - The swin-gpt-2-image-captioning model is trained with the `Trainer` class from the transformers library.
# - The arguments used for training are set via `TrainingArguments` class.
# - The training is started by the `train ` method of the trainer class.

# In[19]:



training_arg = TrainingArguments(
    output_dir='../models/swin_image_captioning',  # The output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=2,  # number of training epochs
    per_device_train_batch_size=64,  # batch size for training
    per_device_eval_batch_size=64,  # batch size for evaluation
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

# In[20]:


trainer.evaluate()

# In[21]:


# 4 epochs took 8 hours
trainer.train()

# In[22]:


# predictions = trainer.predict(processed_dataset['test'])
# for p in predictions.label_ids:
#
#     print(p)
#     # keeping the -100 throws this error --> OverflowError: out of range integral type conversion attempted
#     to_decode = np.where(p != -100, p, tokenizer.pad_token_id)
#     print(to_decode)
#     decoded = tokenizer.decode(to_decode, skip_special_tokens=True)
#     print(decoded)
#     break


# In[34]:


import nltk.translate.bleu_score as bleu

# use logits to convert the prediction

# Calculate the Bleu score for the test set
# predictions = trainer.predict(processed_dataset['test'])
# predicted_label_ids = torch.argmax(predictions.predictions, dim=-1).squeeze().tolist()
# to_decode = [np.where(p != -100, p, tokenizer.pad_token_id) for p in predictions.label_ids]
# predicted_captions = [tokenizer.decode(p, skip_special_tokens=True) for p in to_decode]
#
# def to_np (r):
#     r = np.array(r)
#     r = np.where(r != -100, r, tokenizer.pad_token_id)
#     return r
#
# reference_captions = processed_dataset['test']['labels']
# to_decode_labels = [to_np (r) for r in reference_captions]
# reference_captions_decoded = [tokenizer.decode(l, skip_special_tokens=True) for l in to_decode_labels]
#
#
# bleu_score = bleu.corpus_bleu([[r] for r in reference_captions_decoded], predicted_captions)
#
#
# print("Bleu Score:", bleu_score)


trainer.save_model()

# In[ ]:


# need to save the tokenizer
tokenizer.save_pretrained('../models/swin_image_captioning')

# In[ ]:


# loading model and config from pretrained folder
finetuned_model = VisionEncoderDecoderModel.from_pretrained('../models/swin_image_captioning')

# In[ ]:


"""
Make prediction using the pipeline library
look into wheater you can pass the feature_extractor as a parameter as well.
do_sample: returns multiple options
top_p: getting more confident less random, sharpen predictions

pretrained_generator = pipeline(
    'text-generation', model=model, tokenizer='gpt2',
    config={'max_length': 200, 'do_sample': True, 'top_p': 0.9, 'temperature': 0.7, 'top_k': 10}
)

"""

from IPython.core.display_functions import display

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



