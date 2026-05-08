from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from quantum_cva.amplitude_estimation.experiments.cva import (
    build_6q_cva_problem_bundle,
)
from quantum_cva.amplitude_estimation.experiments.demo import (
    build_demo_problem_bundle,
)
from quantum_cva.amplitude_estimation.experiments.problems import AEProblemBundle


def load_object(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError("Object specs must use 'module:attribute' syntax.")
    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in attr_name.split("."):
        obj = getattr(obj, part)
    return obj


def load_object_from_file(path: str | Path, attr_name: str) -> Any:
    module_path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    obj: Any = module
    for part in str(attr_name).split("."):
        obj = getattr(obj, part)
    return obj


def add_problem_builder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--builder",
        default=None,
        help="Problem builder as 'module:function'. Must return AEProblemBundle.",
    )
    parser.add_argument(
        "--builder-kwargs",
        default="{}",
        help="JSON kwargs passed to --builder.",
    )
    parser.add_argument(
        "--sixq-config",
        default=None,
        help="6q CVA config object as 'module:CONFIG'.",
    )
    parser.add_argument(
        "--sixq-config-path",
        default=None,
        help="Python file containing the 6q CONFIG object.",
    )
    parser.add_argument(
        "--sixq-config-attr",
        default="CONFIG",
        help="Attribute name used with --sixq-config-path.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used to resolve 6q artifact paths.",
    )


def problem_bundle_from_args(args: argparse.Namespace) -> AEProblemBundle:
    if args.sixq_config or args.sixq_config_path:
        if args.sixq_config_path:
            config = load_object_from_file(args.sixq_config_path, args.sixq_config_attr)
        else:
            config = load_object(args.sixq_config)
        return build_6q_cva_problem_bundle(config, repo_root=args.repo_root)

    if args.builder:
        builder = load_object(args.builder)
        kwargs = json.loads(args.builder_kwargs)
        bundle = builder(**kwargs)
        if not isinstance(bundle, AEProblemBundle):
            raise TypeError("--builder must return AEProblemBundle.")
        return bundle

    return build_demo_problem_bundle()
