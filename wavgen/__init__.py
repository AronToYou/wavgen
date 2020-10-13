## Exposes API Components to User ##
import waveform
import constants
import utilities

## Attempts to import the Spectrum drivers' Python header ##
try:
    from .card import Card
except ImportError:
    print("Spectrum drivers missing! Card functions unavailable.")

## Suppresses unnecessary warnings from ##
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="instrumental")  # instrumental deprecation
