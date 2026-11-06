"""Download source dataset to local cache; prints path, does not fetch linked images."""
import kagglehub
from pathlib import Path
p=Path(kagglehub.dataset_download('thedevastator/220k-gpt4vision-image-captions'))/'train.csv'
if not p.is_file():raise FileNotFoundError(p)
print(p)
