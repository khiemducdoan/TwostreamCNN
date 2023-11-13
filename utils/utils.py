import os
import pathlib
def get_class(data_dir):
    classes = sorted(entry.name for entry in os.scandir(data_dir) if entry.is_dir())
    class_to_idx = {class_name:i for i,class_name in enumerate(classes)}
    return classes, class_to_idx