#!/usr/bin/env python
#PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=$PYTHONPATH:./model.py python fine_tune_model_swin_gpt-2.py



# todo modify environmental variable PYTHONPATH

from transformers import TrainingArguments, Trainer
import datasets

try:
    from image_captioning_model.model import ImageCaptioningModel
except ModuleNotFoundError:
    from model import ImageCaptioningModel



# Database
ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir="./dummy_data/")
COCO_DIR = '/data/coco'
# ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=COCO_DIR, split="train[4000:4100]")
# processed_dataset = processed_dataset.train_test_split(test_size=0.05)


# Create instance of the image captioning model
image_captioning_model = ImageCaptioningModel()
image_captioning_model(ds)
del ds
print(image_captioning_model)



# Training Procedure:
# reports evaluation metrics at the end of each epoch on top of the training loss.
training_arg = TrainingArguments(
    output_dir='../models/swin_image_captioning',  # dicts output
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=16,  # training batch size todo try batch size 4
    per_device_eval_batch_size=16,  # evaluation batch size
    load_best_model_at_end=True,
    log_level='info',
    logging_steps=10,
    evaluation_strategy='epoch',  # set to 'steps' and the specify eval_steps=500,. This will give more data to plot
    save_strategy='epoch',
    use_mps_device=True,  # use Apple Silicon
)

trainer = Trainer(
    model=image_captioning_model.model,
    args=training_arg,
    compute_metrics=image_captioning_model.metrics,
    train_dataset=image_captioning_model.processed_dataset['train'],
    eval_dataset=image_captioning_model.processed_dataset['validation'],
    # data_collator=default_data_collator
)

trainer.evaluate()

# Resume fine-tuning from the last checkpoint

# trainer.train(resume_from_checkpoint=True)

# 4 epochs took 8 hours
trainer.train()

# TODO create graphic of metrics per epoch. x axis epoch number and y axis value of the matric
# todo use the test set for evaluation. See chapter 2 ctrl f = preds_output

trainer.save_model()

# save the tokenizer
image_captioning_model.tokenizer.save_pretrained('../models/swin_image_captioning')


