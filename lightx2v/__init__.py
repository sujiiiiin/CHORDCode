__version__ = "0.1.0"
__author__ = "LightX2V Contributors"
__license__ = "Apache 2.0"

# CHORD only vendors `lightx2v` for the tiny-VAE path used by Wan I2V training.
# Keep package init import-light to avoid pulling in the full platform bootstrap.
__all__ = ["__version__", "__author__", "__license__"]
