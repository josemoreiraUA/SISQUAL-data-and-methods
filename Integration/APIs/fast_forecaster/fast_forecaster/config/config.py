import pathlib
import fast_forecaster

PACKAGE_ROOT = pathlib.Path(fast_forecaster.__file__).resolve().parent
DATASET_DIR = PACKAGE_ROOT / 'datasets'
