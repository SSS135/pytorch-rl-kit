import errno
import os
from copy import deepcopy
from pathlib import Path

import torch


class ModelSaver:
    def __init__(self, model_save_folder, model_save_tag=None,
                 model_save_interval=100_000, save_intermediate_models=False, actor_index=0):
        self.model_save_folder = model_save_folder
        self.model_save_interval = model_save_interval
        self.save_intermediate_models = save_intermediate_models
        self.model_save_tag = model_save_tag
        self.actor_index = actor_index
        self._last_model_save_frame = 0

    def check_save_model(self, model, frame):
        if self.model_save_interval is None or self.model_save_folder is None or \
           self._last_model_save_frame + self.model_save_interval > frame:
            return

        self._create_save_folder()
        path = self._get_save_path(frame)
        self._save_model(model, path, frame)

    def _save_model(self, model, path, frame):
        print(f'saving model at {frame} step to {path}')
        model = deepcopy(model).cpu()
        try:
            torch.save(model, path)
        except OSError as e:
            print('error while saving model', e)

    def _get_save_path(self, frame):
        self._last_model_save_frame = frame
        tag = '' if self.model_save_tag is None else self.model_save_tag
        if self.save_intermediate_models:
            name = f'{tag}_{self.actor_index}_{frame}'
        else:
            name = f'{tag}_{self.actor_index}'
        return Path(self.model_save_folder) / (name + '.pth')

    def _create_save_folder(self):
        try:
            os.makedirs(self.model_save_folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise