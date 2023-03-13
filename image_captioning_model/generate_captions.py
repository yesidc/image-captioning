from transformers import VisionEncoderDecoderModel
from image_captioning_model.model import GenerateCaptions


# loading model and config from pretrained folder
finetuned_model = VisionEncoderDecoderModel.from_pretrained('../models/swin_image_captioning')
generate_captions = GenerateCaptions()






captions, model_output, img_transformed = generate_captions(  # Out of sample photo
    finetuned_model, '../data/test_data/000000421195_test.jpg'
)
print(captions)