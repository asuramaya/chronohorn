"""Evaluation harnesses — chronohorn orchestration over decepticons kernel organs.

Promotes the one-off ``experiments/harnesses/knn_stream*.py`` scripts into
first-class package architecture: the census-style state-kNN datastore eval,
composed from the kernel primitives (``LinearStateStreamer`` +
``StateKNNMemory``) rather than re-inlining FFT streaming, whitening, tiled
search, and the marginal backoff by hand.
"""
from .knn_datastore import KNNDatastoreConfig, run_knn_datastore

__all__ = ["KNNDatastoreConfig", "run_knn_datastore"]
