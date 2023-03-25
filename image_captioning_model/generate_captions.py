# PYTHONPATH=$PYTHONPATH:./model.py  python generate_captions.py
import logging
import time
from transformers import VisionEncoderDecoderModel
import datasets

try:
    from image_captioning_model.model import GenerateCaptions
except ModuleNotFoundError:
    from model import GenerateCaptions

# create logger
logger = logging.getLogger('image_captioning')


def generate_captions_and_evaluate(path_to_finetuned_model, path_to_data, evaluate=False):
    # loading model and config from pretrained folder
    finetuned_model = VisionEncoderDecoderModel.from_pretrained(path_to_finetuned_model)
    generate_captions = GenerateCaptions(finetuned_model)
    if evaluate:
        ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=path_to_data)
        ds = ds['validation']
        start_time = time.time()
        logger.info(f'COMPUTING METRICS FOR: {path_to_data}')
        generate_captions.evaluate_predictions(ds)
        end_time = time.time()
        logger.info(f'TIME ELAPSED: {end_time - start_time}')
        logger.info(f'METRICS COMPUTED: {generate_captions.results}')
    else:
        captions = generate_captions.generate_caption(path_to_data, evaluate)
        print(captions)


# generate_captions_and_evaluate(path_to_finetuned_model='/Users/yesidcano/Documents/SWIN-GPT/swin-no-F-GPT', path_to_data='../data/test_data/images')
generate_captions_and_evaluate(path_to_finetuned_model='/Users/yesidcano/Documents/SWIN-GPT/swin-no-F-GPT',
                               path_to_data='/Users/yesidcano/repos/image-captioning/data/coco', evaluate=True)

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
