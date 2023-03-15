#PYTHONPATH=$PYTHONPATH:./model.py  python generate_captions.py

from transformers import VisionEncoderDecoderModel

try:
    from image_captioning_model.model import GenerateCaptions
except ModuleNotFoundError:
    from model import GenerateCaptions



# loading model and config from pretrained folder
finetuned_model = VisionEncoderDecoderModel.from_pretrained('../models/swin_image_captioning')
generate_captions = GenerateCaptions()

captions, model_output, img_transformed = generate_captions(
    finetuned_model, '../data/test_data/000000421195_test.jpg'
)
print(captions)
