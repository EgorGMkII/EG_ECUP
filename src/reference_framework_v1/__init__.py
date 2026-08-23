"""Config-driven temporal experiment framework.

The package deliberately wraps ``ssl_temporal_stack_v1`` instead of replacing
its frozen entrypoint and result contract.
"""

from .config import ExperimentConfig, load_experiment_config
from .profiles import POST_NY_PUBLIC_PROXY, TemporalProfile

__all__ = ["ExperimentConfig", "POST_NY_PUBLIC_PROXY", "TemporalProfile", "load_experiment_config"]
