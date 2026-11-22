"""Named, independent random streams derived from one integer seed.

Every noise source in the simulator draws from its own stream, keyed by a path of names
(e.g. ``("surface", 0)`` or ``("camera", "deformed", 3)``). Streams are independent of one
another and of the order in which they are used, so adding a new noise source never
changes the realisation of an existing one.

NumPy does not guarantee bit-identical ``Generator`` output across NumPy versions, so
tests should assert statistical properties or tolerances rather than exact pixel values.
"""

from __future__ import annotations

import zlib

import numpy as np


def _key(name: object) -> int:
    return zlib.crc32(str(name).encode("utf-8"))


def stream(seed: int, *names: object) -> np.random.Generator:
    """Return a generator for the stream identified by ``seed`` and ``names``."""
    seq = np.random.SeedSequence(entropy=seed, spawn_key=tuple(_key(n) for n in names))
    return np.random.Generator(np.random.PCG64(seq))
