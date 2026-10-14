from pathlib import Path
import shutil
import requests
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
URL = ("https://github.com/StackExchange/Survey/raw/refs/heads/main/"
       "packages/archive/{year}/results.csv")


def download_data(year: int) -> Path:
    """Download one year's survey CSV into data/ if not already present."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"survey_{year}.csv"
    if not path.exists():
        r = requests.get(URL.format(year=year), timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path

def load_years(years = (2023,  2024, 2025)) -> dict:
    """Return {year: DataFrame} for the requested survey years."""
    return {y: pd.read_csv(download_data(y)) for y in years}

def harmonize(frames: dict) -> pd.DataFrame:
    """Keep only columns shared by all years and stack them with a Year column."""
    common = sorted(set.intersection(*[set(df.columns) for df in frames.values()]))
    return pd.concat(
        [frames[y][common].assign(Year=y) for y in frames.keys()],
        ignore_index=True
    )

def make_raw_archive() -> Path:
   """Zip data/ for the Google Drive deliverable (zip is gitignored)."""
   out = PROJECT_ROOT / "stackoverflow_survey_raw_2023_2025" 
   return Path(shutil.make_archive(str(out), "zip", DATA_DIR))


