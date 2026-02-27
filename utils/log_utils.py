import logging
import os

def configure_cache_logger(name: str, file_name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        os.makedirs("./logs", exist_ok=True)
        handler = logging.FileHandler(f"./logs/{file_name}")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(handler)

    return logger