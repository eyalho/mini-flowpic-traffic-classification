import logging
from pathlib import Path
import zipfile

DATA_FOLDER_PATH = Path('data')
DS_ZIP_FILENMAE = Path('ucdavis_quic.zip')
DS_PATH = DATA_FOLDER_PATH / DS_ZIP_FILENMAE.stem
DS_ZIP_PATH = DATA_FOLDER_PATH / DS_ZIP_FILENMAE

logging.info(f"Extracting {DS_ZIP_PATH} to {DATA_FOLDER_PATH}")
if not DS_PATH.exists():
    with zipfile.ZipFile(DS_ZIP_PATH, 'r') as unzip_obj:
        try:
            unzip_obj.extractall(DATA_FOLDER_PATH)
            logging.info(f"finish Extracting")
        except FileExistsError as e:
            logging.warning(f"{e}")
assert DS_PATH.exists()
