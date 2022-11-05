import logging
import pathlib
from datetime import datetime as date


PARENT_PATH = str(pathlib.Path().parent.resolve())

logging_path = PARENT_PATH + "\\log"
logger = logging.getLogger()
info = logger.info
debug = logger.debug
warning = logger.warning

def log_config(name) : 
    logging.basicConfig(
        level = logging.DEBUG,
        format = " {levelname:<8} {asctime} {message}",
        style='{',
        filename=logging_path + f'\\{name}_{date.today().strftime("%d-%m-%Y_%Hh%M")}.log',
        filemode='w'
    )
