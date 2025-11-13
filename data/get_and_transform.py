import kagglehub
import os
import pandas as pd
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_latest_version() -> str:
    path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")
    print("Path to dataset files:", path)
    return path

def build_metadata_path(path: str) -> str:
    return os.path.join(path, "movies_metadata.csv")

def load_metadata_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        low_memory=False,
        usecols=["original_title", "overview", "original_language"],
    )
    df = df.rename(columns={"original_title": "title"})
    return df

def clean_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    # Clean and filter rows
    df = df.dropna(subset=["title", "overview", "original_language"]).copy()
    df["title"] = df["title"].astype(str).str.strip()
    df["overview"] = df["overview"].astype(str).str.strip()
    df["original_language"] = df["original_language"].astype(str).str.strip()
    df = df[(df["title"] != "") & (df["overview"] != "") & (df["original_language"] != "")]
    return df

def ensure_output_dir() -> Path:
    out_dir = Path("data/texts")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def sanitize_filename(name: str, used: set, max_len: int = 100) -> str:
    name = re.sub(r"[\\/*?:\"<>|]", "_", name)
    name = re.sub(r"\s+", " ", name).strip().rstrip(". ")
    base = name[:max_len]
    candidate = base
    i = 1
    # Ensure uniqueness (case-insensitive) and avoid clobbering existing files
    while candidate.lower() in used or (Path("data/texts") / f"{candidate}.txt").exists():
        i += 1
        suffix = f"_{i}"
        candidate = (base[: max_len - len(suffix)] + suffix)
    used.add(candidate.lower())
    return candidate

def prepare_write_items(df: pd.DataFrame, out_dir: Path):
    # Precompute filenames sequentially to avoid thread-safety issues
    used_names = set()
    items = []  # list[(Path, str)]
    for _, row in df.iterrows():
        fname = sanitize_filename(row["title"], used_names)
        content = (
            f"Film Title: {row['title']}\n"
            f"Descritpion: {row['overview']}\n"
            f"original language: {row['original_language']}\n"
        )
        items.append((out_dir / f"{fname}.txt", content))
    return items

def write_file(item):
    p, content = item
    p.write_text(content, encoding="utf-8")
    return True

def write_all(items) -> int:
    written = 0
    max_workers = min(32, (os.cpu_count() or 1) * 5)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(write_file, it) for it in items]
        for fut in as_completed(futures):
            try:
                if fut.result():
                    written += 1
            except Exception:
                pass
    return written

def main():
    path = download_latest_version()
    movies_metadata_path = build_metadata_path(path)
    df = load_metadata_df(movies_metadata_path)
    df = clean_and_filter(df)
    out_dir = ensure_output_dir()
    items = prepare_write_items(df, out_dir)
    written = write_all(items)
    print(f"Wrote {written} documents to {out_dir}")

if __name__ == "__main__":
    main()
