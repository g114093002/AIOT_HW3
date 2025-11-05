This folder contains a small sample of the SMS Spam dataset and a downloader
script to fetch the canonical full CSV on demand.

- `sms_spam_no_header_sample.csv` : A tiny sample (safe to keep in git).
- `download_dataset.py` : Script to download the full dataset into this
  directory. Run `python dataset/download_dataset.py --out sms_spam_no_header.csv`.

Why this structure?
- Large raw datasets often bloat git history. The sample is useful for quick
  local testing and CI. Use the downloader to fetch the real CSV only when
  needed.

If you prefer I can add the full CSV to the repository, or commit it in a
separate large-file branch (or use Git LFS). Tell me which you prefer.
