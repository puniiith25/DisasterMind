from fastapi import FastAPI

app = FastAPI(title="DisasterMind")

@app.get("/")
def root():
    return {"message":"DisasteMind API Server Running"}