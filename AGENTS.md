# AGENTS.md

Project-specific notes for AI coding agents working in `utils3d`. For library
purpose and the full function inventory see [README.md](README.md).

## Project shape

- Pure-Python library of standalone 3D vision/graphics functions.
- Two parallel backends: [utils3d/numpy/](utils3d/numpy/) and
  [utils3d/torch/](utils3d/torch/). Most public functions exist in both.
- [utils3d/interface/](utils3d/interface/) is **auto-generated** and dispatches
  `utils3d.foo(...)` to the numpy or torch implementation based on input type
  (see `_call_based_on_args` in [gen_interface.py](gen_interface.py)).
- Short aliases: `utils3d.np` ↔ `utils3d.numpy`, `utils3d.pt` ↔ `utils3d.torch`.

## House style (non-negotiable — see [README.md](README.md))

- **Flat & non-modular**: standalone functions only, no classes/hierarchies
  (existing exceptions: `RastContext`).
- **Vectorized only**: no Python loops beyond O(log N) over data.
- **Native types only**: plain Python / `numpy.ndarray` / `torch.Tensor`. No
  custom wrapper types.
- **Backend parity**: when adding/changing a function in one backend, mirror
  it in the other unless there's a clear reason (e.g. nvdiffrast-only).
  Keep function name, parameter names, and parameter order identical across
  backends so the auto-dispatcher works.

## How a public function gets wired up

1. Implement in `utils3d/{numpy,torch}/<module>.py` (e.g. `transforms.py`).
2. Add the name to that file's top-level `__all__` list. The package init
   uses `lazy_import_all_from` ([utils3d/helpers.py](utils3d/helpers.py))
   keyed off `__all__`.
3. Regenerate metadata + dispatcher + docs (see next section).

The `_<module>.__all__.py` files (e.g.
[utils3d/numpy/_transforms.__all__.py](utils3d/numpy/_transforms.__all__.py))
and [utils3d/interface/__init__.py](utils3d/interface/__init__.py) /
[utils3d/interface/__init__.pyi](utils3d/interface/__init__.pyi) are
**auto-generated — do not hand-edit**.

## Required regeneration commands

Run from the repo root after any change to a public function's existence,
name, or signature:

```bash
python gen_interface.py   # regenerates utils3d/interface/__init__.py(i)
python gen_doc.py         # regenerates _doc.md (paste into README.md by hand)
```

Importing `utils3d` after editing `__all__` will auto-refresh the matching
`_<module>.__all__.py` meta file (with a `LazyImportWarning`); running
`gen_interface.py` also exercises this path.

## Decorators on torch/numpy functions

Torch implementations frequently stack
`@totensor(_others=torch.float32)` then `@batched(_others=0)`
([utils3d/torch/helpers.py](utils3d/torch/helpers.py)). Numpy uses
`@toarray` + `@batched` ([utils3d/numpy/helpers.py](utils3d/numpy/helpers.py)).
These handle dtype coercion and leading-batch-dim broadcasting — match the
existing pattern in neighbouring functions rather than hand-rolling
broadcasting logic.

## Things that look like cruft but aren't

- `utils3d/*/_*.__all__.py` — meta files for the lazy-import system. Keep.
- `build/` and `utils3d.egg-info/` — build artefacts; ignore.
- `test.py` is an ad-hoc PLY-I/O benchmark, not a test suite. There is no
  formal test runner; verify changes with a small script or REPL.

## Install / dev

```bash
pip install -e .
```

Dependencies (from [pyproject.toml](pyproject.toml)): `numpy`, `scipy`,
`moderngl`. Torch and `nvdiffrast` are **not** installed automatically but are
required for `utils3d.torch.*` and the rasterization paths respectively.
