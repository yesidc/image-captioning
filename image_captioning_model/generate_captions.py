# PYTHONPATH=$PYTHONPATH:./model.py  python generate_captions.py
import logging
import time
from transformers import VisionEncoderDecoderModel
from helpers import load_dataset

try:
    from image_captioning_model.model import GenerateCaptions
except ModuleNotFoundError:
    from model import GenerateCaptions

# create logger
logger = logging.getLogger('image_captioning')


def generate_captions_and_evaluate(path_to_finetuned_model, path_to_data, evaluate=False, dummy_data=False):
    # loading model and config from pretrained folder
    finetuned_model = VisionEncoderDecoderModel.from_pretrained(path_to_finetuned_model)
    generate_captions = GenerateCaptions(finetuned_model)
    if evaluate:
        ds = load_dataset(path_to_data, dummy_data=dummy_data)
        ds = ds['validation']
        start_time = time.time()
        logger.info(f'COMPUTING METRICS using dataset: {ds}. Transformed images will be cached.')
        generate_captions.evaluate_predictions(ds,batch_size=70)
        end_time = time.time()
        logger.info(f'TIME ELAPSED: {end_time - start_time}')
        logger.info(f'METRICS COMPUTED: {generate_captions.results}')
    else:
        captions = generate_captions.generate_caption(path_to_data)
        print(captions)


# generate_captions_and_evaluate(path_to_finetuned_model='/Users/yesidcano/Documents/SWIN-GPT/swin-no-F-GPT', path_to_data='../data/test_data/images')
generate_captions_and_evaluate(path_to_finetuned_model='/Users/yesidcano/Documents/SWIN-GPT/swin-no-F-GPT',
                               path_to_data='../data/test_data/images', evaluate=False, dummy_data=False)

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
