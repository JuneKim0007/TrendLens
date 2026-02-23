from pathlib import Path

file_path = Path(__file__).resolve().parent
db_path = str(file_path.parents[0] / 'Database')

from .data_loader import DataLoader
from .cache import AnalysisCache