# Alphabetagama AI Group Finder

This is a proof-of-concept Streamlit application that recommends the five most relevant Viva Engage groups for a user's interest.

## Files

- `app.py` - Streamlit application
- `dataset.xlsx` - synthetic Viva Engage dataset
- `requirements.txt` - Python packages required by the app
- `.streamlit/config.toml` - light theme configuration
- `.gitignore` - files that should not be committed to Git

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The first run downloads the `sentence-transformers/all-MiniLM-L6-v2` model. This is expected.

## Important production note

The included Excel file contains synthetic data for demonstration only. Do not upload real employee emails, private group information or Viva Engage posts to a public GitHub repository. Use a private repository and an approved secure data source for real data.
