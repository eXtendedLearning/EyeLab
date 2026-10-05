"""EyeLab validation: printable ground-truth plates, headless capture, offline analysis.

Protocol and rationale: ``docs/validation.md``. Run from ``python/``::

    python -m validation --help

Coupling to EyeLab is deliberate and narrow:

- ``capture`` drives ``pose_estimator.ArucoPipeline`` and ``overlay.project_nodes``
  exactly as the GUI does, and records what they produce;
- ``target`` reuses ``registration.marker_object_corners`` and
  ``registration.save_marker_config`` so marker geometry has one definition.

Everything else reads the recorded session format only. Units follow the repo
invariant: metres internally, millimetres only in fields named ``*_mm``.
"""
