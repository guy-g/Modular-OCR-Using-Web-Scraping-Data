import logging
import os


def create_log(running_name, running_folder, log_file_path=None):
    if log_file_path is None:
        log_file_path = os.path.join(running_folder, 'log.txt')
    if os.path.isfile(log_file_path):
        os.remove(log_file_path)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(running_name)
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

