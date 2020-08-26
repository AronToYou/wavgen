## Exposes API Components to User ##
from .config import *
from .utilities import Wave, Step, from_file, rp, plot_waveform, plot_ends

## Attempts to import the Spectrum drivers' Python header ##
try:
    from .card import Card
except ImportError:
    print("Spectrum drivers missing! Card functions unavailable.")

## Suppresses unnecessary warnings from ##
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="instrumental")  # instrumental deprecation

## Imports all of the User Defined Waveforms ##
# Retrieves a list of all waveforms
import importlib
import inspect
wavs = inspect.getmembers(importlib.import_module(f"{__name__}.waveform"), inspect.isclass)
funcs = inspect.getmembers(importlib.import_module(f"{__name__}.waveform"), inspect.isfunction)

# Find the Base Class
for i, (name, _) in enumerate(wavs):
    if name == 'Waveform':
        _, Waveform = wavs.pop(i)
        break

# Import all extensions of the Base Class
for name, wav in wavs:
    if issubclass(wav, Waveform):
        globals()[name] = wav

# Import all functions from waveform
for name, func in funcs:
    globals()[name] = func

# Remove the Base Class from the global namespace
del Waveform
