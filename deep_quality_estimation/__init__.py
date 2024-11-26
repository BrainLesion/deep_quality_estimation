from loguru import logger

logger.disable("deep_quality_estimation")  # disable for use a library

from .model import DQE
