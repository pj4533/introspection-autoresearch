"""Model lifecycle registry.

A `ModelHandle` is the contract: explicit `load()` / `unload()` /
`is_loaded()` for any model the four-phase worker needs to move in and out
of memory. The contract is deliberately minimal so that PyTorch-MPS Gemma
and MLX-format judges/proposers both fit, and a `MockHandle` is available
for tests and integration smoke runs.

Crucial invariant: only ONE handle is loaded at a time across the whole
process. The worker enforces this — handles do not police each other. But
each `unload()` must release its Metal/MPS allocations cleanly, otherwise
a leak across swaps will OOM the next load.

`enforce_free_memory()` is the safety belt: call it before any `load()`
to refuse to start a load that we already know won't fit.
"""

from __future__ import annotations

import gc
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


# ----------------------------------------------------------------------
# Memory introspection helpers
# ----------------------------------------------------------------------

def free_memory_gb() -> float:
    """Return free physical memory in GB on macOS.

    Uses `vm_stat` because psutil's `available` on macOS reports a number
    that ignores compressed memory and macOS's adaptive cache, which is
    not what we want for "can I fit a 35 GB model right now". `vm_stat`'s
    "Pages free" + "Pages inactive" + "Pages purgeable" is the closest
    proxy to actual free RAM that's available to a new allocation.
    """
    try:
        out = subprocess.check_output(["vm_stat"], text=True, timeout=2)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return -1.0  # unknown — let caller decide whether to proceed

    # Parse page size from the header line, e.g.:
    #   "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
    # macOS 14+ on Apple Silicon uses 16 KB pages; fall back to 4 KB.
    import re
    page_size = 4096
    m = re.search(r"page size of\s+(\d+)\s+bytes", out)
    if m:
        page_size = int(m.group(1))

    free_pages = 0
    for line in out.splitlines():
        parts = line.split(":")
        if len(parts) != 2:
            continue
        key, val = parts[0].strip(), parts[1].strip().rstrip(".").replace(",", "")
        if key in ("Pages free", "Pages inactive", "Pages purgeable"):
            try:
                free_pages += int(val)
            except ValueError:
                pass
    return (free_pages * page_size) / (1024 ** 3)


def enforce_free_memory(
    min_gb: float, *,
    raise_on_unknown: bool = False,
    retry_seconds: float = 5.0,
) -> None:
    """Raise if free memory is below `min_gb`. No-op if free memory is unknown.

    Kernel page reclaim runs lazily — vm_stat may report stale "free" pages
    for several seconds after a large unload. We retry once after
    `retry_seconds` so a healthy unload-then-load doesn't trip the gate just
    because the kernel hasn't caught up.
    """
    import time as _time
    free = free_memory_gb()
    if free < 0:
        if raise_on_unknown:
            raise RuntimeError("could not determine free memory; refusing to load")
        return
    if free < min_gb and retry_seconds > 0:
        _time.sleep(retry_seconds)
        free = free_memory_gb()
    if free < min_gb:
        raise RuntimeError(
            f"insufficient free memory: {free:.1f} GB free, need {min_gb:.1f} GB. "
            "Likely cause: a previous unload didn't fully release its allocations. "
            "Investigate before retrying — silent activation corruption is the failure mode."
        )


# ----------------------------------------------------------------------
# Handle contract
# ----------------------------------------------------------------------

class ModelHandle(ABC):
    """Lifecycle contract for any model the worker swaps in and out.

    Subclasses implement `_do_load()` and `_do_unload()`. The base class
    handles the load/unload state and a basic free-memory pre-check.

    `obj` is whatever the runtime needs to use the loaded model — for
    Gemma it's a DetectionPipeline, for MLX models it's a (model, tokenizer)
    tuple. Callers introspect type via the concrete handle subclass.
    """

    name: str
    expected_ram_gb: float

    def __init__(self, name: str, *, expected_ram_gb: float = 0.0):
        self.name = name
        self.expected_ram_gb = expected_ram_gb
        self._loaded: bool = False
        self._obj: Any = None

    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def obj(self) -> Any:
        if not self._loaded:
            raise RuntimeError(f"{self.name} is not loaded")
        return self._obj

    def load(self, *, free_gb_required: Optional[float] = None) -> None:
        if self._loaded:
            return
        # Headroom: assume 4 GB OS reserve on top of the model's footprint.
        # The pre-check is a sanity gate against silent OOM, not a precise
        # predictor — vm_stat's "free" reading lags the kernel's actual
        # reclaim, and torch's MPS cache holds buffers across unloads.
        # A too-strict gate fires spuriously on a healthy run.
        required = (
            free_gb_required
            if free_gb_required is not None
            else (self.expected_ram_gb + 4.0)
        )
        if required > 0:
            enforce_free_memory(required)
        self._obj = self._do_load()
        self._loaded = True

    def unload(self) -> None:
        if not self._loaded:
            return
        try:
            self._do_unload(self._obj)
        finally:
            self._obj = None
            self._loaded = False
            gc.collect()

    @abstractmethod
    def _do_load(self) -> Any:
        ...

    @abstractmethod
    def _do_unload(self, obj: Any) -> None:
        ...

    def __repr__(self) -> str:
        state = "loaded" if self._loaded else "unloaded"
        return f"<{type(self).__name__} {self.name} [{state}]>"


# ----------------------------------------------------------------------
# Concrete handles
# ----------------------------------------------------------------------

class GemmaHandle(ModelHandle):
    """Wraps the existing src.bridge / src.inject DetectionPipeline.

    Loading installs the bf16 model on MPS, derives no per-concept vector
    yet (that happens per-candidate later). Unloading attempts to release
    Metal allocations via `torch.mps.empty_cache()` — anything left
    untouched leaks across the swap.

    Lazy imports torch/transformers so importing this module is cheap
    (e.g. for tests of the Mock or MLX handles).
    """

    def __init__(
        self,
        *,
        model_id: str = "gemma3_12b",
        name: str = "gemma3_12b",
        expected_ram_gb: float = 24.0,
        abliteration_path: Optional[Path] = None,
    ):
        super().__init__(name=name, expected_ram_gb=expected_ram_gb)
        self.model_id = model_id
        self.abliteration_path = abliteration_path

    def _do_load(self) -> Any:
        # Import lazily — torch + transformers + paper code is a heavy import.
        from src.bridge import DetectionPipeline, load_gemma_mps

        model = load_gemma_mps(self.model_id)
        pipeline = DetectionPipeline(model=model)
        if self.abliteration_path is not None:
            from src.paper.abliteration import AbliterationContext
            ctx = AbliterationContext.from_file(
                self.abliteration_path, pipeline.model
            )
            ctx.install()
            pipeline.abliteration_ctx = ctx
        return pipeline

    def _do_unload(self, obj: Any) -> None:
        # Remove abliteration hooks (if any) before tearing down model.
        ctx = getattr(obj, "abliteration_ctx", None)
        if ctx is not None:
            try:
                ctx.remove()
            except Exception:
                pass
        # Drop large attributes to make sure references die.
        for attr in ("model", "tokenizer", "abliteration_ctx"):
            if hasattr(obj, attr):
                try:
                    setattr(obj, attr, None)
                except Exception:
                    pass
        # Best-effort MPS cache reclaim. PyTorch's MPS allocator caches
        # buffers across Python-side reference drops; an explicit
        # synchronize + empty_cache is the documented escape hatch. Doing
        # it twice tends to give the kernel time to actually reclaim.
        try:
            import torch
            if hasattr(torch, "mps"):
                if hasattr(torch.mps, "synchronize"):
                    torch.mps.synchronize()
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                    torch.mps.empty_cache()
        except ImportError:
            pass


class MLXHandle(ModelHandle):
    """Wraps an MLX-format model loaded via `mlx_lm.load`.

    `obj` is a `(model, tokenizer)` tuple. Sub-tools like the local judge
    and proposer pass this tuple into `mlx_lm.generate` directly.

    Unload semantics: `mlx_lm` doesn't expose a public unload; we drop our
    references and call `mx.metal.clear_cache()` (best-effort) to free
    Metal heap. If MLX leaks, the next handle's `enforce_free_memory()`
    pre-check will catch it.
    """

    def __init__(
        self,
        *,
        model_path: str,
        name: Optional[str] = None,
        expected_ram_gb: float = 0.0,
    ):
        super().__init__(
            name=name or Path(model_path).name,
            expected_ram_gb=expected_ram_gb,
        )
        self.model_path = model_path

    def _do_load(self) -> Any:
        try:
            from mlx_lm import load  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "mlx-lm not installed. Run: pip install -U mlx mlx-lm"
            ) from e
        return load(self.model_path)

    def _do_unload(self, obj: Any) -> None:
        # `obj` is (model, tokenizer). MLX models hold parameters as
        # mx.array Metal buffers; calling .clear() on the dict returned by
        # model.parameters() doesn't free them because that dict is a flat
        # view, not the source of truth. The actual buffers are stored on
        # the modules themselves. Replace every parameter with an empty
        # mx.array so the originals lose their last reference.
        try:
            model, _tokenizer = obj
            try:
                import mlx.core as mx  # type: ignore
                from mlx.utils import tree_flatten, tree_unflatten  # type: ignore
                flat = tree_flatten(model.parameters())
                empty_flat = [(k, mx.zeros((1,))) for k, _ in flat]
                model.update(tree_unflatten(empty_flat))
            except Exception:
                pass
        except (TypeError, ValueError):
            pass

        try:
            import mlx.core as mx  # type: ignore
            # Synchronize first so any in-flight ops complete and release.
            if hasattr(mx, "synchronize"):
                try:
                    mx.synchronize()
                except Exception:
                    pass
            # MLX 0.31+: top-level mx.clear_cache + mx.reset_peak_memory.
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            if hasattr(mx, "reset_peak_memory"):
                mx.reset_peak_memory()
        except ImportError:
            pass


class MockHandle(ModelHandle):
    """In-memory mock for tests and integration smoke without real weights.

    `obj` is a user-supplied value (or a dict by default). `load`/`unload`
    just toggle state and run optional callbacks. Useful for exercising
    the worker state machine without paying minutes-per-load cost.
    """

    def __init__(
        self,
        *,
        name: str = "mock",
        load_value: Any = None,
        on_load=None,
        on_unload=None,
        expected_ram_gb: float = 0.0,
    ):
        super().__init__(name=name, expected_ram_gb=expected_ram_gb)
        self._load_value = load_value if load_value is not None else {"name": name}
        self._on_load = on_load
        self._on_unload = on_unload
        self.load_count = 0
        self.unload_count = 0

    def _do_load(self) -> Any:
        self.load_count += 1
        if self._on_load is not None:
            self._on_load()
        return self._load_value

    def _do_unload(self, obj: Any) -> None:
        self.unload_count += 1
        if self._on_unload is not None:
            self._on_unload()


@dataclass
class HandleRegistry:
    """Holds the worker's two model handles (gemma, judge) and provides
    a one-at-a-time guarantee: switching to a new active handle unloads
    all others first.

    Phase 2g dropped the third (proposer) slot — SAE features come from
    Neuronpedia, not from an LLM proposer, so Phase C is pure CPU work.
    """

    gemma: ModelHandle
    judge: ModelHandle

    def all(self) -> list[ModelHandle]:
        return [self.gemma, self.judge]

    def loaded(self) -> list[ModelHandle]:
        return [h for h in self.all() if h.is_loaded()]

    def activate(self, handle: ModelHandle) -> ModelHandle:
        """Make `handle` the only loaded handle in this registry.

        Logs every unload + load with elapsed time and free-memory readings
        so the worker log surfaces exactly when the model swaps happen and
        how long they took. The Monitor key-greps these `[swap]` lines.
        """
        import time as _time
        if handle not in self.all():
            raise ValueError(
                f"handle {handle.name!r} is not registered in this registry"
            )
        for h in self.all():
            if h is handle:
                continue
            if h.is_loaded():
                t0 = _time.time()
                print(
                    f"[swap] unloading {h.name} (free_gb={free_memory_gb():.1f}) ...",
                    flush=True,
                )
                h.unload()
                print(
                    f"[swap] unloaded  {h.name} in {_time.time()-t0:.1f}s "
                    f"(free_gb={free_memory_gb():.1f})",
                    flush=True,
                )
        if not handle.is_loaded():
            t0 = _time.time()
            print(
                f"[swap] loading   {handle.name} (free_gb={free_memory_gb():.1f}) ...",
                flush=True,
            )
            handle.load()
            print(
                f"[swap] loaded    {handle.name} in {_time.time()-t0:.1f}s "
                f"(free_gb={free_memory_gb():.1f})",
                flush=True,
            )
        return handle

    def unload_all(self) -> None:
        for h in self.all():
            if h.is_loaded():
                h.unload()
