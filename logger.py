import os
import logging

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "log")
LOG_FILE = os.path.join(LOG_DIR, "baseline.log")


class Logger:
    @classmethod
    def initialize(cls, log_file=LOG_FILE):
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
        cls.logger = logging.getLogger(__name__)
        cls.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(filename)s:: %(message)s"
        )
        stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(os.path.join(log_file))
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        cls.logger.addHandler(stream_handler)
        cls.logger.addHandler(file_handler)

    @classmethod
    def get_logger(cls):
        try:
            return cls.logger
        except:
            cls.initialize()
            return cls.logger
