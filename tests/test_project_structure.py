"""
tests/test_project_structure.py — Directory layout and gitignore verification tests.
"""
from pathlib import Path

ROOT = Path(r"d:\canvas\NIDS")

REQUIRED_DIRS = [
    "data/raw", "data/raw/cic-ids-2017", "data/raw/lycos-ids-2017",
    "data/raw/unsw-nb15", "data/raw/ctu-13", "data/processed",
    "data/lab_captures", "data/splits", "src/data", "src/models",
    "src/evaluation", "src/explainability", "src/response", "src/drift",
    "src/dashboard", "src/dashboard/components", "src/utils",
    "configs", "experiments", "notebooks", "tests", "lab",
    "lab/attack_scripts", "lab/vm_configs", "lab/capture_logs",
    "results", "results/figures", "docs"
]

REQUIRED_PACKAGE_INITS = [
    "src", "src/data", "src/models", "src/evaluation",
    "src/explainability", "src/response", "src/drift",
    "src/dashboard", "src/dashboard/components", "src/utils"
]

def test_all_directories_exist():
    """Verify all required project directories exist."""
    missing = [d for d in REQUIRED_DIRS if not (ROOT / d).is_dir()]
    assert not missing, f"Missing directories: {missing}"

def test_package_inits_exist():
    """Verify __init__.py exists in all Python packages."""
    missing = [p for p in REQUIRED_PACKAGE_INITS if not (ROOT / p / "__init__.py").is_file()]
    assert not missing, f"Missing __init__.py in: {missing}"

def test_gitignore_exists_and_configured():
    """Verify .gitignore exists and properly ignores heavy/data directories."""
    gitignore_path = ROOT / ".gitignore"
    assert gitignore_path.is_file(), ".gitignore file missing"
    content = gitignore_path.read_text()
    
    assert "data/raw/" in content or "data/raw" in content, ".gitignore should exclude data/raw/"
    assert ".venv" in content, ".gitignore should exclude .venv"
    assert "planning" not in content.splitlines(), ".gitignore MUST NOT ignore planning directory"

def test_requirements_exists():
    """Verify requirements.txt exists."""
    assert (ROOT / "requirements.txt").is_file(), "requirements.txt missing"
