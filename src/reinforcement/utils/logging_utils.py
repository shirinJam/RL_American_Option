import subprocess
import logging
import sys
import resource
import platform


def initialise_logger(log_level=logging.DEBUG, name="root"):
    """
    Initialise logger.
    :param log_level: log level
    :param name: logger name
    :return: logger
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.propagate = False

        logger.setLevel(log_level)

        # add handler to stderr
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.warning(
            "Logger initialised with level {}".format(logging.getLevelName(log_level))
        )

    return logger
