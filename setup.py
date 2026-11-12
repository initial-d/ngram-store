"""Build the standalone C reader; wheels have no Python C-API dependency.

`tools/` holds the runtime modules and the experiment scripts behind `evidence/`.
Wheels ship only the runtime: the import closure of the entry points below.
"""
import ast
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


ROOT = Path(__file__).resolve().parent
# Modules reachable from the public API, CLI and console scripts.
RUNTIME_ENTRY_POINTS = {
    "native_row_store", "block_compression", "text_classifier", "budgeted_bags",
    "certified_torch", "certified_store", "external_cotraining", "cotraining_groups",
    "structural_native", "engram_row_adapter", "device_row_adapter", "mapped_embedding_bag",
    "tinyengram_checkpoint", "fasttext_native_bundle", "prepare_fasttext_corpus",
    "verify_tinyengram_checkpoint", "verify_tinyengram_device", "train_tinyengram_reference",
    "demo_certified_store", "quantized_memory",
}


def runtime_modules():
    """Import closure of RUNTIME_ENTRY_POINTS within tools/."""
    available = {p.stem for p in (ROOT / "tools").glob("*.py")}
    def imports(name):
        found = set()
        for node in ast.walk(ast.parse((ROOT / "tools" / (name + ".py")).read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    found.add(parts[1] if parts[0] in ("tools", "meft_memory") and len(parts) > 1 else parts[0])
                    if alias.name.startswith("meft_memory._runtime.") and len(parts) > 2:
                        found.add(parts[2])
            elif isinstance(node, ast.ImportFrom):
                parts = (node.module or "").split(".")
                found.add(parts[-1])
                if node.module in (None, "tools", "meft_memory._runtime"):
                    found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.add(node.value.rsplit(".", 1)[-1])   # import_module("...name")
        return found & available
    closure, pending = set(), set(RUNTIME_ENTRY_POINTS)
    while pending:
        name = pending.pop()
        if name not in closure:
            closure.add(name)
            pending |= imports(name) - closure
    return closure


class BuildNative(build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if package != "meft_memory._runtime":
            return modules
        keep = runtime_modules() | {"__init__"}
        return [m for m in modules if m[1] in keep]

    def run(self):
        if sys.platform not in ("darwin", "linux"):
            raise RuntimeError("native row-store wheels currently support Linux and macOS")
        super().run()
        directory = Path(self.build_lib) / "meft_memory" / "_native"
        directory.mkdir(parents=True, exist_ok=True)
        darwin = sys.platform == "darwin"
        filename = "libmeft_rows.dylib" if darwin else "libmeft_rows.so"
        compiler = shlex.split(os.environ.get("CC", "cc"))
        command = [*compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC",
                   "-dynamiclib" if darwin else "-shared", str(Path(__file__).parent / "native" / "meft_rows.c"),
                   "-lz", "-o", str(directory / filename)]
        # Explicit opt-in keeps portable wheels independent of libdeflate.
        # The linked library must be available on the deployment machine.
        if os.environ.get("MEFT_USE_LIBDEFLATE") == "1":
            command.extend(["-DMEFT_USE_LIBDEFLATE", "-ldeflate"])
        for codec, library in (("LZ4", "lz4"), ("ZSTD", "zstd")):
            if os.environ.get("MEFT_USE_" + codec) == "1":
                command.extend(["-DMEFT_USE_" + codec, "-l" + library])
        command.extend(shlex.split(os.environ.get("MEFT_NATIVE_FLAGS", "")))
        if darwin:
            command.extend(["-arch", platform.machine()])
        subprocess.run(command, check=True)


class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, platform_tag = super().get_tag()
        if sys.platform == "darwin":
            # Universal2 Python does not make a plain cc-built dylib universal.
            # Compile/tag the running architecture; never advertise both.
            platform_tag = "_".join(platform_tag.split("_")[:3]) + "_" + platform.machine()
        return "py3", "none", platform_tag


setup(cmdclass={"build_py": BuildNative, "bdist_wheel": PlatformWheel})
