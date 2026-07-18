"""MOVED (2026-07-09): folded into chronohorn.eval.knn_datastore.

The stage-3c windowed-states + key-transform-arm sweep is now the module's
`--states-mode windowed` + `--arm` path, composing the same decepticons kernel
organ (StateKNNMemory) it always used. The windowed-state extraction and the
arm loop live in the module now; nothing here to keep.

    python -m chronohorn.eval.knn_datastore --states-mode windowed --arm pca128 jl128 pca64

This shim forwards the arm sweep to the module.
"""
import sys

from chronohorn.eval.knn_datastore import KNNDatastoreConfig, run_knn_datastore

if __name__ == "__main__":
    arms = sys.argv[1:] or ["pca128", "jl128"]
    for arm in arms:
        run_knn_datastore(KNNDatastoreConfig.from_arm(arm, states_mode="windowed"))
