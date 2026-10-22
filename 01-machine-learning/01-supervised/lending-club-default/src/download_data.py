"""Step 1 (Extract): download the Lending Club loan data from Kaggle.

Needs a Kaggle API token in ~/.kaggle/kaggle.json (Kaggle > Settings > API >
Create New Token). The download is about 1.7 GB; we keep only the accepted-loans
file (about 375 MB compressed) and delete the rest.

The data is not committed to git (see .gitignore): anyone can rebuild the
project by running this script.

Run from the project folder:  python src/download_data.py
"""
import shutil
import subprocess
import sys
import zipfile

from config import KAGGLE_DATASET, RAW_DIR, RAW_PATTERN


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(RAW_DIR.rglob(RAW_PATTERN))
    if existing:
        print(f"Already downloaded: {existing[0].relative_to(RAW_DIR)}")
        return

    if shutil.which("kaggle") is None:
        sys.exit("The Kaggle CLI is missing. Run `pip install kaggle`, then try again.")

    print(f"Downloading {KAGGLE_DATASET} (about 1.7 GB, this takes a few minutes)...")
    subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(RAW_DIR)],
                   check=True)

    archive_path = next(RAW_DIR.glob("*.zip"))
    with zipfile.ZipFile(archive_path) as archive:
        members = [m for m in archive.namelist() if "accepted_2007_to_2018" in m]
        # Prefer the compressed copy: same data, 5x smaller on disk.
        member = next((m for m in members if m.endswith(".gz")), members[0])
        print(f"Extracting {member}")
        target = RAW_DIR / member.split("/")[-1]
        with archive.open(member) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    archive_path.unlink()
    print(f"Saved {target.relative_to(RAW_DIR.parent.parent)} ({target.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
