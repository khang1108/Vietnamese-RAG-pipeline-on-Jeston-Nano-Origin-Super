import os 
import sys
from loguru import logger as lg

LOG_DIR = "logs" 
os.makedirs(LOG_DIR, exist_ok=True)
log_filepath = os.path.join(LOG_DIR, "log.logs")

lg.remove()

lg.add(
    sys.stderr, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG" 
)

lg.add(
    log_filepath,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO", 
    rotation="10 MB", 
    retention="7 days", 
    enqueue=True 
)

lg.level("WARNING", color="<yellow><bold>") 
lg.level("ERROR", color="<red><bold><underline>") 

logger = lg