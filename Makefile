setup:
	python -m pip install -r requirements.txt

download:
	python scripts/download_data.py

run:
	python -m src.pipeline run --category bottle

train:
	python -m src.pipeline train --category bottle

evaluate:
	python -m src.pipeline evaluate --category bottle

serve:
	uvicorn src.api:app --host 0.0.0.0 --port 8000

test:
	python -m pytest -q
