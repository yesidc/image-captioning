# PYTHONPATH=$PYTHONPATH:./model.py  python generate_captions.py
import logging
import time
from transformers import VisionEncoderDecoderModel
from helpers import load_dataset

try:
    from image_captioning_model.model import GenerateEvaluateCaptions
except ModuleNotFoundError:
    from model import GenerateEvaluateCaptions

# create logger
logger = logging.getLogger('image_captioning')


def generate_captions_and_evaluate(path_to_finetuned_model,
                                   path_to_data=None,
                                   validation_data=None,
                                   dataset_type=None,
                                   evaluate=False,
                                   dummy_data=False):
    """

    :param path_to_finetuned_model:
    :param path_to_data:
    :param validation_data:
    :param evaluate:
    :param dummy_data:
    :return:
    """
    # loading model and config from pretrained folder
    finetuned_model = VisionEncoderDecoderModel.from_pretrained(path_to_finetuned_model)
    generate_captions = GenerateEvaluateCaptions(finetuned_model)
    if evaluate:
        if path_to_data:
            ds = load_dataset(PATH_DATASET=path_to_data, dummy_data=dummy_data,dataset_type=dataset_type)
            ds = ds['validation']
        elif validation_data:
            ds = validation_data
        else:
            raise ValueError('Either path_to_data or validation_data must be provided')
        start_time = time.time()
        logger.info(f'COMPUTING METRICS using dataset: {ds}. Transformed images will be cached.')
        generate_captions.evaluate_predictions(ds,batch_size=70)
        end_time = time.time()
        logger.info(f'TIME ELAPSED: {end_time - start_time}')
        logger.info(f'METRICS COMPUTED: {generate_captions.results}')
        return generate_captions.results
    else:
        captions = generate_captions.generate_caption(path_to_data)
        print(captions)

#
# generate_captions_and_evaluate(path_to_finetuned_model='/Users/yesidcano/Library/Mobile Documents/com~apple~CloudDocs/models-tuned/swin-gpt-0-2-frozen/checkpoint-236704',
#                                path_to_data='/Users/yesidcano/repos/image-captioning/data/coco',
#                                dataset_type='coco',
#                                evaluate=True)

# generate_captions_and_evaluate(path_to_finetuned_model='/Users/yesidcano/Documents/SWIN-GPT/swin-no-F-GPT',
#                                path_to_data='../data/test_data/images', evaluate=False, dummy_data=False)

#path_to_data='/Users/yesidcano/repos/image-captioning/data/coco'

# # loading model and config from pretrained folder

# image_to_text = pipeline("image-to-text", model="/Users/yesidcano/Documents/SWIN-GPT/swin-no-F-GPT")
#
# generate_kwargs = {
#     "num_return_sequences":3,
#     #  "num_beams":3,
#     # 'top_k': 10,
#     # 'early_stopping': True,
#     # 'max_length':15,
# }

# transformers==4.23.1
#
# from transformers import pipeline, AutoFeatureExtractor
#
#
# image_to_text = pipeline("image-to-text", model="../models/swin_image_captioning")
#
# generate_kwargs = {
#     "num_return_sequences":3,
#      "num_beams":3,
#     'top_k': 10,
#     'early_stopping': True,
#     'max_length':15,
# }
#
#
#
# p = image_to_text("C:\\Users\\yesid\\Pictures\\black.png", generate_kwargs=generate_kwargs)
# print()
