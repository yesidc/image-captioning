#PYTHONPATH=$PYTHONPATH:./model.py  python generate_captions.py

from transformers import VisionEncoderDecoderModel
from transformers import pipeline, AutoFeatureExtractor

try:
    from image_captioning_model.model import GenerateCaptions
except ModuleNotFoundError:
    from model import GenerateCaptions



# loading model and config from pretrained folder
finetuned_model = VisionEncoderDecoderModel.from_pretrained("/Users/yesidcano/Documents/SWIN-GPT/swin-no-F-GPT")
generate_captions = GenerateCaptions(finetuned_model)

captions = generate_captions.generate_caption('../data/test_data/images'
)
print(captions)


image_to_text = pipeline("image-to-text", model="/Users/yesidcano/Documents/SWIN-GPT/swin-no-F-GPT")

generate_kwargs = {
    "num_return_sequences":3,
    #  "num_beams":3,
    # 'top_k': 10,
    # 'early_stopping': True,
    # 'max_length':15,
}



p = image_to_text('/Users/yesidcano/repos/image-captioning/data/test_data/zebras-black-white-7324423.jpg', generate_kwargs=generate_kwargs)
print()


