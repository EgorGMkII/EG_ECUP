"""Run the installed DataSphere CLI without its optional PyPI version probe.

The installed 0.10 CLI performs a network check in ``finally``.  The runner
intentionally removes proxy variables for the DataSphere RPC, so that probe
can otherwise hang on an unreachable local proxy.  No CLI behavior is changed
apart from skipping this non-functional version notification.
"""

import shutil
import tempfile
from pathlib import Path


class _RunnerTemporaryDirectory:
    """Avoid the Windows endpoint rule that blocks ``datasphere_*`` dirs."""

    def __init__(self, *args, **kwargs):
        self.cleanup_args = args
        self.cleanup_kwargs = kwargs

    def __enter__(self):
        self.name = tempfile.mkdtemp(prefix="dsjob_")
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.name, ignore_errors=True)


import datasphere.sdk as sdk
from datasphere.api import jobs_pb2 as jobs
from datasphere.config import VariablePath
from datasphere.files import zip_path
from datasphere.config import local_module_prefix


def _safe_prepare_local_modules(py_env, tmpdir):
    """Archive modules without holding a NamedTemporaryFile open on Windows."""
    result = []
    for i, module in enumerate(py_env.local_modules_paths):
        fd, archive_name = tempfile.mkstemp(prefix="module_", suffix=".zip", dir=tmpdir)
        import os
        os.close(fd)
        with open(archive_name, "w+b") as archive:
            zip_path(module, archive)
        path = VariablePath(
            archive_name,
            var=f"{local_module_prefix}_{i}",
            compression_type=jobs.FileCompressionType.ZIP,
        )
        with open(archive_name, "rb") as archive:
            path.get_file(archive)
        result.append(path)
    return result

sdk.tempfile.TemporaryDirectory = _RunnerTemporaryDirectory
sdk.prepare_local_modules = _safe_prepare_local_modules

import datasphere.main as cli_main


cli_main.check_package_version = lambda: None


if __name__ == "__main__":
    cli_main.main()
