import logging

LOG_FORMAT = '%(asctime)s  %(levelname)5s  %(message)s'


def get_logger(name: str, log_level: str = 'info', log_format: str = None):
    if log_format is None:
        log_format = LOG_FORMAT
    logging.basicConfig(level=log_level.upper(), format=log_format)
    logger = logging.getLogger(name)
    logger.addHandler(get_console_handler(log_level, log_format))
    logger.propagate = False # otherwise root logger prints things again
    return logger


def get_file_handler(log_file: str, log_level: str = 'info', log_format: str = None)-> logging.FileHandler:
    if log_format is None:
        log_format = LOG_FORMAT
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level.upper())
    file_handler.setFormatter(logging.Formatter(log_format))
    return file_handler


def get_console_handler(log_level: str = 'info', log_format: str = None)-> logging.StreamHandler:
    if log_format is None:
        log_format = LOG_FORMAT
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level.upper())
    console_handler.setFormatter(logging.Formatter(log_format))
    return console_handler