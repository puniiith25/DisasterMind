# Topic 14: Docker Deployment & Container Orchestration

## 1. Objective
To deliver containerized microservices deployment for the entire DisasterMind platform:
- FastAPI Backend API (`disastermind-backend`)
- Local Qdrant Vector Store (`disastermind-qdrant`)
- Disaster Assessment React UI Dashboard (`disastermind-frontend`)

---

## 2. Docker Compose Configuration

The unified deployment stack is defined in `docker-compose.yml`:
- Port `8000`: FastAPI API and Swagger docs.
- Port `6333`: Qdrant REST vector search.
- Port `3000`: React UI web interface.

---

## 3. Deployment Commands

### Build & Start Stack
```bash
docker compose up --build -d
```

### Check Logs & Health Status
```bash
docker compose ps
docker compose logs -f disastermind-backend
```

### Stop Stack
```bash
docker compose down
```
