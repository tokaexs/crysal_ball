
from pathlib import Path
import hashlib

DATA_DIR = Path(__file__).resolve().parent

# Keep the original files; only scan generated/export folders.
SCAN_DIRS = [
    DATA_DIR / "csv_exports",
]

def file_hash(path):
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def find_duplicates(folder):
    seen = {}
    duplicates = []

    if not folder.exists():
        return duplicates

    for path in folder.rglob("*"):
        if not path.is_file():
            continue

        digest = file_hash(path)

        if digest in seen:
            duplicates.append((path, seen[digest]))
        else:
            seen[digest] = path

    return duplicates


def main():
    duplicates = []

    for folder in SCAN_DIRS:
        duplicates.extend(find_duplicates(folder))

    if not duplicates:
        print("No exact duplicate files found.")
        return

    print("\nExact duplicates found:\n")

    for duplicate, original in duplicates:
        print(f"Duplicate: {duplicate}")
        print(f"Keeping:   {original}\n")

    confirm = input(
        "Type DELETE to permanently remove these duplicate files: "
    )

    if confirm != "DELETE":
        print("Cancelled. No files were deleted.")
        return

    for duplicate, original in duplicates:
        duplicate.unlink()
        print(f"Deleted: {duplicate}")

    print("\nDone! Duplicate files removed.")


if __name__ == "__main__":
    main()