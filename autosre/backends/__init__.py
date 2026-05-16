"""Backend implementations for different platforms."""

from .base import Backend, BackendType, detect_platform
from .llamacpp import LlamaCppBackend
from .mlx_dflash import MlxDflashBackend
from .ollama import OllamaBackend
from .vllm import VllmBackend

__all__ = [
    "Backend",
    "BackendType",
    "LlamaCppBackend",
    "MlxDflashBackend",
    "OllamaBackend",
    "VllmBackend",
    "detect_platform",
    "get_backend",
]


def get_backend(
    backend_type: BackendType | str | None = None,
    active_state: dict[str, object] | None = None,
) -> Backend:
    """Get the appropriate backend for the current platform or specified type.

    Args:
        backend_type: Which backend to use. None = auto-detect.
        active_state: Previously saved state from active.json. Passed to the
            backend constructor for reconstruction after process restart.
            Required for vLLM backend to know which nodes/containers to manage.
    """
    if backend_type is None:
        backend_type = detect_platform()

    if isinstance(backend_type, str):
        backend_type = BackendType(backend_type)

    backends: dict[BackendType, type[Backend]] = {
        BackendType.OLLAMA: OllamaBackend,
        BackendType.LLAMACPP: LlamaCppBackend,
        BackendType.VLLM: VllmBackend,
        BackendType.MLX_DFLASH: MlxDflashBackend,
    }

    return backends[backend_type](active_state=active_state)
