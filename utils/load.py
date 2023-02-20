from pathlib import Path
from config.default import default_export_path, model_to_save_file_type
import glob

from model.overlay import CLModel

def try_load(export_path:str, load_model:str|None):
    model = None
    export_path = Path(export_path) if export_path is not None else Path(default_export_path)
    Path.mkdir(export_path, parents=True, exist_ok=True, mode=model_to_save_file_type)
    if(load_model is not None):
        find_path = export_path / load_model
        path = glob.glob(str(find_path), recursive=True)
        if('checkpoint' in load_model):
            if('CLModel' in load_model):
                model = CLModel.load_from_checkpoint(path)
                print(f'INFO: Loaded model "{path}"')
    return model