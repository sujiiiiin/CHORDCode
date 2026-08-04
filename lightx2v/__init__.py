from pkgutil import extend_path


# CHORD vendors only a subset of LightX2V. Allow a pinned upstream installation
# to supply missing modules (currently the Tiny-VAE implementation) without
# replacing CHORD's local chord_adapter and runtime files.
__path__ = extend_path(__path__, __name__)

__version__ = "0.1.0"
__author__ = "LightX2V Contributors"
__license__ = "Apache 2.0"

# CHORD only vendors `lightx2v` for the tiny-VAE path used by Wan I2V training.
# Keep package init import-light to avoid pulling in the full platform bootstrap.
__all__ = ["__version__", "__author__", "__license__"]
