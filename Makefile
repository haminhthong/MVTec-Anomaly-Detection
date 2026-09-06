CATEGORY ?= bottle
DATA_ROOT ?= data/raw
MODEL_ROOT ?= models
REPORT_ROOT ?= reports

setup:
	python -m pip install -r requirements.txt

download:
	python scripts/download_data.py

train:
	python -m src.pipeline --stage train --category $(CATEGORY) --data-root $(DATA_ROOT) --model-root $(MODEL_ROOT)

evaluate:
	python -m src.pipeline --stage evaluate --category $(CATEGORY) --data-root $(DATA_ROOT) --model-root $(MODEL_ROOT) --report-root $(REPORT_ROOT)

pipeline:
	python -m src.pipeline --stage all --category $(CATEGORY) --data-root $(DATA_ROOT) --model-root $(MODEL_ROOT) --report-root $(REPORT_ROOT)

serve:
	uvicorn src.api:app --host 0.0.0.0 --port 8000

test:
	pytest -q
