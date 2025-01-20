import json
import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler


class ColorCodes:
    GREY = "\x1b[38;21m"
    BLUE = "\x1b[38;5;39m"
    YELLOW = "\x1b[38;5;226m"
    RED = "\x1b[38;5;196m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"


class ColorFormatter(logging.Formatter):
    def __init__(self, fmt: str):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: ColorCodes.GREY + fmt + ColorCodes.RESET,
            logging.INFO: ColorCodes.BLUE + fmt + ColorCodes.RESET,
            logging.WARNING: ColorCodes.YELLOW + fmt + ColorCodes.RESET,
            logging.ERROR: ColorCodes.RED + fmt + ColorCodes.RESET,
            logging.CRITICAL: ColorCodes.BOLD_RED + fmt + ColorCodes.RESET,
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class JsonFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        return dt.isoformat()

    def format(self, record):
        json_record = {
            "message": record.getMessage(),
            "level": record.levelname,
            "timestamp": self.formatTime(record),
        }
        if record.exc_info:
            json_record["stacktrace"] = self.formatException(record.exc_info)
        return json.dumps(json_record)


def get_logger(stage: str = "dev") -> logging.Logger:
    """
    :param stage:
    :return:
    """
    name = "app"
    log_dir = "logs"
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.debug("Logger already configured")
        return logger

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if stage == "dev":
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColorFormatter(log_format))
        logger.addHandler(console_handler)
    else:
        logger.setLevel(logging.INFO)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = TimedRotatingFileHandler(
            filename=os.path.join(log_dir, f"{name}.log"), when="midnight", interval=1, backupCount=30, encoding="utf-8"
        )
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
    return logger
