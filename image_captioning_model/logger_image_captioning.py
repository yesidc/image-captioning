import logging

# create logger
logger = logging.getLogger('image_captioning')
logger.setLevel(logging.DEBUG)

# create file handler which logs messages to a file
fh = logging.FileHandler('swin_F_1_GPT_E_5_image_captioning.log')
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(filename)s %(funcName)s  %(lineno)d: %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


