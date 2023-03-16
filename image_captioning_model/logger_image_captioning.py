import logging

# create logger
logger = logging.getLogger('image_captioning')
logger.setLevel(logging.DEBUG)

# create file handler which logs messages to a file
fh = logging.FileHandler('pipeline_image_caption.log')
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('[%(levelname)s] %(filename)s %(funcName)s  %(lineno)d: %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


