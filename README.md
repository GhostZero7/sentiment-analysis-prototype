# Sentiment Analysis Prototype

## Step 1 Implemented: Apify Data Collection

This stage includes Facebook URL validation and comment collection through Apify.

### Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy environment variables:

```bash
cp .env.example .env
```

4. Edit `.env` and set `APIFY_API_KEY`.

### Run Data Download

```bash
python scripts/download_data.py --url "https://www.facebook.com/..."
```

Output will be written to `data/raw/comments.csv` by default.
