"""MOVED (2026-07-09): promoted to chronohorn.eval.knn_datastore.

The round-3 census state-kNN recipe is now package architecture, composed from
the decepticons kernel primitives (LinearStateStreamer for continuous-stream
states + StateKNNMemory with truncate-then-whiten keys, tiled search, and the
marginal backoff) instead of re-inlining them here. The module was verified
BIT-FOR-BIT against this former script (base 2.0367, kNN-alone 2.9743, mix delta
-0.0144 +/-0.0122, 18/32) before this shim replaced it.

Store size and the RAM tier are now config, not a copy-pasted file:

    python -m chronohorn.eval.knn_datastore --store-bytes 8000000
    python -m chronohorn.eval.knn_datastore --store-bytes 32000000 --store-device cpu

This shim forwards to the module so the old invocation still runs.
"""
from chronohorn.eval.knn_datastore import main

if __name__ == "__main__":
    raise SystemExit(main())
