python3 -m venv venv

source venv/bin/activate

pip install fastapi uvicorn python-multipart python-dotenv

uvicorn app.main:app --reload

pip install -U datasets huggingface_hub

hf download Kingdrone-Junjue/DisasterM3 
  DisasterM3_Bench/benchmark_release.json 
  --repo-type dataset 
  --local-dir data/disasterm3

pip install -U qdrant-client sentence-transformers
