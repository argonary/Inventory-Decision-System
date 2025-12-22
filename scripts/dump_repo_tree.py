from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".idea",
    ".pytest_cache",
    ".mypy_cache",
    ".DS_Store",
}

EXCLUDE_FILES = {
    ".DS_Store",
}

def print_tree(path: Path, prefix: str = ""):
    entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))

    for i, entry in enumerate(entries):
        if entry.name in EXCLUDE_DIRS or entry.name in EXCLUDE_FILES:
            continue

        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry.name)

        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(entry, prefix + extension)

if __name__ == "__main__":
    print(f"\n📁 Project tree for: {PROJECT_ROOT.name}\n")
    print(PROJECT_ROOT.name)
    print_tree(PROJECT_ROOT)
