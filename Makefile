# Makefile for ERP Anomaly Detector

.PHONY: setup train run-api run-dashboard test clean lint help

help:
	@echo "Available commands:"
	@echo "  setup         - Initialize directories and copy local artifact files"
	@echo "  train         - Run Random Forest training pipeline to save model files"
	@echo "  run-api       - Launch FastAPI backend locally"
	@echo "  run-dashboard - Launch Streamlit dashboard app locally"
	@echo "  test          - Run full pytest test suite"
	@echo "  clean         - Clean python cache files and logs"

setup:
	python scripts/setup_artifacts.py

train:
	python ml/training/train.py

run-api:
	python backend/main.py

run-dashboard:
	streamlit run frontend/dashboard.py

test:
	pytest tests/ -v

clean:
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
