#PYTHONPATH=$PYTHONPATH:./model.py  python generate_captions.py

# from transformers import VisionEncoderDecoderModel
#
# try:
#     from image_captioning_model.model import GenerateCaptions
# except ModuleNotFoundError:
#     from model import GenerateCaptions
#
#
#
# # loading model and config from pretrained folder
# finetuned_model = VisionEncoderDecoderModel.from_pretrained('../models/swin_image_captioning')
# generate_captions = GenerateCaptions()
#
# captions, model_output, img_transformed = generate_captions(
#     finetuned_model, '../data/test_data/000000421195_test.jpg'
# )
# print(captions)

# OSError: ../models/swin_image_captioning does not appear to have a file named preprocessor_config.json.

#transformers==4.23.1

from transformers import pipeline, AutoFeatureExtractor


image_to_text = pipeline("image-to-text", model="../models/swin_image_captioning")

generate_kwargs = {
    "num_return_sequences":3,
     "num_beams":3,
    'top_k': 10,
    'early_stopping': True,
    'max_length':15,
}



p = image_to_text('/Users/yesidcano/repos/image-captioning/data/test_data/zebras-black-white-7324423.jpg', generate_kwargs=generate_kwargs)
print()


