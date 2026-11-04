import importlib.util
import sys
from pathlib import Path


EXTERNAL_REPOS_ROOT = Path(
    "/Net/Groups/BGI/scratch/crobin/PythonProjects/external_repos"
)


def load_external_class(
    repo_name: str,
    module_path: str,
    class_name: str = "Model",
    module_name: str | None = None,
):
    """
    Load a model class from an external research repository without
    replacing this project's own `models` package.

    Parameters
    ----------
    repo_name:
        Directory under external_repos/.

    module_path:
        Path relative to repository root, e.g.
        "models/TimeXer.py".

    class_name:
        Class to load. Defaults to "Model", which is the convention
        used by the TimeXer repository.

    module_name:
        Unique Python module name. If omitted, it is generated from
        repo_name and module_path.
    """

    repo_root = EXTERNAL_REPOS_ROOT / repo_name
    source_file = repo_root / module_path

    if not repo_root.exists():
        raise FileNotFoundError(f"External repository not found: {repo_root}")

    if not source_file.exists():
        raise FileNotFoundError(f"External model file not found: {source_file}")

    if module_name is None:
        safe_repo = repo_name.lower().replace("-", "_")
        safe_module = source_file.stem.lower().replace("-", "_")

        module_name = f"external_{safe_repo}_{safe_module}"

    repo_root_str = str(repo_root)

    # External repositories commonly use imports such as:
    #
    #     from layers.Embed import ...
    #
    # so expose the repository root only while loading.
    sys.path.insert(0, repo_root_str)

    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            source_file,
        )

        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {source_file}")

        module = importlib.util.module_from_spec(spec)

        sys.modules[module_name] = module

        spec.loader.exec_module(module)

        try:
            return getattr(module, class_name)
        except AttributeError as exc:
            raise ImportError(f"{class_name!r} not found in {source_file}") from exc

    finally:
        # Do not leave the external repository on sys.path.
        if repo_root_str in sys.path:
            sys.path.remove(repo_root_str)
