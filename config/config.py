import json
import os

class Config:
    def __init__(self):
        path = os.path.join("config", "vars.json")
        with open(path) as f:
            self.cfg = json.load(f)

    def __getitem__(self, key):
        return self.cfg[key]

config = Config()
