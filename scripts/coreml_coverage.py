"""Predict how a model will be partitioned by ONNX Runtime's CoreML (MLProgram) EP, on any OS.

Runs onnxruntime.tools.check_onnx_model_mobile_usability with one correction: the Python
checker treats empty optional inputs (e.g. Resize's absent `roi`, input name '') as tensors of
unknown shape, whereas ORT's C++ partitioner ignores absent optional inputs. Without the fix
every Resize is wrongly reported as "dynamic shape" and splits the partition.

Usage: python scripts/coreml_coverage.py <model.onnx>   (needs `onnx` installed)
"""
from __future__ import annotations

import builtins
import logging
import pathlib
import sys

from onnxruntime.tools.mobile_helpers import usability_checker as uc

_all = builtins.all


def _all_ignoring_absent_inputs(iterable):
    frame = sys._getframe(1)
    if frame.f_code.co_name == "_check_partitioning_for_graph" and "node" in frame.f_locals:
        fixed = frame.f_locals["_is_fixed_shape_value"]
        return _all(fixed(i) for i in frame.f_locals["node"].input if i)
    return _all(iterable)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    uc.all = _all_ignoring_absent_inputs  # shadows the builtin inside usability_checker only
    uc.analyze_model(pathlib.Path(sys.argv[1]), skip_optimize=False, logger=logging.getLogger("coreml"))


if __name__ == "__main__":
    main()
