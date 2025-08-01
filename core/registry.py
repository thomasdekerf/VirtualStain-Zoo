
import importlib, pkgutil, inspect, pathlib, sys
REGISTRY = {}

def register(cls):
    REGISTRY[cls.__name__.lower()] = cls
    return cls

def discover_models():
    models_path = pathlib.Path(__file__).resolve().parent.parent/'models'
    for py in models_path.glob('*.py'):
        if py.stem != '__init__':
            importlib.import_module(f'models.{py.stem}')

def get_model(name, **kwargs):
    discover_models()
    key = name.lower()
    if key not in REGISTRY:
        raise ValueError(f'Model {name} not found in registry. Available: {list(REGISTRY.keys())}')
    return REGISTRY[key](kwargs)

