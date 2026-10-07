python3 -m venv venv

source venv/bin/activate



pip install fastapi uvicorn python-multipart python-dotenv

uvicorn app.main:app --reload