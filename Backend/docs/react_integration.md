# Topic 12: React Frontend Dashboard Integration

## 1. Objective
To provide an interactive, operator-grade disaster assessment console connecting directly to DisasterMind's FastAPI backend:
- Interactive RAG query builder with preset disaster scenarios (Earthquake, Flood, Tornado).
- Multi-temporal visual assessment preview.
- Grounded prediction display with citation links and veracity score.
- Real-time live disaster bulletins poll via USGS GeoJSON integration.
- Dataset validation and split distributions dashboard.

---

## 2. Architecture & Tech Stack

- **Framework**: React 18 + Vite
- **Styling**: Vanilla CSS glassmorphism system with curated dark palette
- **Port**: `3000` (proxied to Backend `8000`)

---

## 3. Running Frontend Locally

```bash
cd Frontend
npm install
npm run dev
```

The console opens at [http://localhost:3000](http://localhost:3000).
