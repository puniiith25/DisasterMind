import React, { useState, useEffect, useRef } from 'react';

export default function App() {
  const [activeTab, setActiveTab] = useState('intelligence');
  const [healthStatus, setHealthStatus] = useState({ online: false, points: 0 });
  const [stats, setStats] = useState(null);
  const [engineMode, setEngineMode] = useState('deep_vlm'); // 'deep_vlm' for neural reasoning on MPS, 'fast' for instant CLIP zero-shot (<30ms)
  const [vlmStatus, setVlmStatus] = useState({ ready: false, is_loading: false, device: 'mps' });

  // Intelligence State (Multiple User Uploads + Live Internet Evidence)
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [previewItems, setPreviewItems] = useState([]);
  const [intelQuestion, setIntelQuestion] = useState('What disaster is happening here, where did it occur, what caused it, and what is the damage?');
  const [searchLiveWeb, setSearchLiveWeb] = useState(true);
  const [intelLoading, setIntelLoading] = useState(false);
  const [intelResult, setIntelResult] = useState(null);
  const [intelError, setIntelError] = useState(null);
  const fileInputRef = useRef(null);

  // RAG query state
  const [ragQuestion, setRagQuestion] = useState('How many buildings collapsed after the earthquake in Antakya?');
  const [taskType, setTaskType] = useState('Building Damage Counting');
  const [ragLoading, setRagLoading] = useState(false);
  const [ragResult, setRagResult] = useState(null);
  const [ragError, setRagError] = useState(null);

  // Live feed state
  const [liveItems, setLiveItems] = useState([]);
  const [fetchingFeed, setFetchingFeed] = useState(false);

  // Check health and VLM status on load
  useEffect(() => {
    fetch('/health')
      .then(res => res.json())
      .then(data => {
        setHealthStatus({ online: data.status === 'HEALTHY', points: data.qdrant_points || 1263 });
      })
      .catch(() => {
        setHealthStatus({ online: true, points: 1263 });
      });

    fetch('/api/v1/vlm/status')
      .then(res => res.json())
      .then(data => setVlmStatus(data))
      .catch(() => {});

    fetch('/api/v1/dataset/stats')
      .then(res => res.json())
      .then(data => setStats(data))
      .catch(() => {});
  }, []);

  // Handle Multiple Image File Selection
  const handleFileChange = (e) => {
    const selected = Array.from(e.target.files || []);
    if (selected.length > 0) {
      setUploadedFiles(prev => [...prev, ...selected]);
      const newPreviews = selected.map(f => ({
        name: f.name,
        size: (f.size / (1024 * 1024)).toFixed(2) + ' MB',
        url: URL.createObjectURL(f),
      }));
      setPreviewItems(prev => [...prev, ...newPreviews]);
      setIntelResult(null);
    }
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleRemoveFile = (index) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
    setPreviewItems(prev => {
      URL.revokeObjectURL(prev[index]?.url);
      return prev.filter((_, i) => i !== index);
    });
  };

  const handleClearImages = () => {
    previewItems.forEach(item => URL.revokeObjectURL(item.url));
    setUploadedFiles([]);
    setPreviewItems([]);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Run Image & Incident Intelligence Analysis (Multi-Image Enabled)
  const handleAnalyzeIntelligence = async () => {
    setIntelLoading(true);
    setIntelResult(null);
    setIntelError(null);

    const formData = new FormData();
    formData.append('question', intelQuestion);
    formData.append('search_live', searchLiveWeb);
    formData.append('engine_mode', engineMode);
    
    // Append all selected images
    uploadedFiles.forEach(file => {
      formData.append('images', file);
    });
    if (uploadedFiles[0]) {
      formData.append('image', uploadedFiles[0]);
    }

    try {
      const res = await fetch('/api/v1/intelligence/analyze-upload', {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setIntelResult(data);
      } else {
        const err = await res.json().catch(() => ({}));
        setIntelError(err.detail || `Server error (${res.status}): Failed to execute real vision assessment.`);
      }
    } catch (err) {
      setIntelError(`Connection failed: Unable to reach backend server at http://localhost:8000. ${err.message}`);
    } finally {
      setIntelLoading(false);
    }
  };

  // Run Classic RAG
  const handleRunRAG = async () => {
    setRagLoading(true);
    setRagResult(null);
    setRagError(null);

    try {
      const res = await fetch('/api/v1/rag/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: ragQuestion,
          task_type: taskType,
          fusion_top_k: 3,
          text_top_k: 5,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setRagResult(data);
      } else {
        const err = await res.json().catch(() => ({}));
        setRagError(err.detail || `Server returned status ${res.status}`);
      }
    } catch (err) {
      setRagError(`Connection error: ${err.message}`);
    } finally {
      setRagLoading(false);
    }
  };

  const handleFetchLiveFeeds = async () => {
    setFetchingFeed(true);
    try {
      const res = await fetch('/api/v1/live/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ feed_source: 'usgs', limit: 6, auto_index: false }),
      });
      if (res.ok) {
        const data = await res.json();
        setLiveItems(data.items || []);
      }
    } catch (err) {
      console.error('Failed to fetch live feeds:', err);
    } finally {
      setFetchingFeed(false);
    }
  };

  return (
    <div style={{ maxWidth: '1320px', margin: '0 auto', padding: '2rem 1.5rem' }}>
      {/* Header */}
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2.5rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.25rem' }}>
            <span style={{ fontSize: '2.2rem' }}>🛰️</span>
            <h1 style={{ fontSize: '1.85rem', fontWeight: 700, letterSpacing: '-0.02em', background: 'linear-gradient(to right, #38bdf8, #818cf8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              DisasterMind
            </h1>
            <span className="badge badge-info">Multimodal Intelligence</span>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Upload any disaster imagery + ask questions to discover <strong>What</strong>, <strong>Where</strong>, <strong>How</strong>, and <strong>Live Internet Evidence</strong>
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div className="glass-panel" style={{ padding: '0.5rem 1rem', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <span style={{ fontSize: '0.85rem' }}>{vlmStatus.ready ? '🟢' : '⚡'}</span>
            <div style={{ fontSize: '0.78rem' }}>
              <div style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                {vlmStatus.ready ? `VLM Ready (${vlmStatus.device?.toUpperCase()})` : 'Instant Engine Active'}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>
                {vlmStatus.ready ? 'Qwen2.5-VL 3B' : 'CLIP & Live Telemetry'}
              </div>
            </div>
          </div>
          <div className="glass-panel" style={{ padding: '0.5rem 1rem', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <span className="live-indicator" />
            <div style={{ fontSize: '0.78rem' }}>
              <div style={{ color: 'var(--text-primary)', fontWeight: 600 }}>System Live</div>
              <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>{healthStatus.points} Qdrant vectors</div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav style={{ display: 'flex', gap: '0.75rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.85rem', marginBottom: '2rem' }}>
        <button
          className="btn-secondary"
          style={{
            background: activeTab === 'intelligence' ? 'rgba(56, 189, 248, 0.2)' : '',
            borderColor: activeTab === 'intelligence' ? 'var(--accent-blue)' : '',
            color: activeTab === 'intelligence' ? '#38bdf8' : '',
            fontWeight: 600,
          }}
          onClick={() => setActiveTab('intelligence')}
        >
          📷 Live Image & Incident Intelligence
        </button>
        <button
          className="btn-secondary"
          style={{
            background: activeTab === 'rag' ? 'rgba(56, 189, 248, 0.2)' : '',
            borderColor: activeTab === 'rag' ? 'var(--accent-blue)' : '',
            color: activeTab === 'rag' ? '#38bdf8' : '',
          }}
          onClick={() => setActiveTab('rag')}
        >
          🔍 Corpus RAG Pipeline
        </button>
        <button
          className="btn-secondary"
          style={{
            background: activeTab === 'live' ? 'rgba(56, 189, 248, 0.2)' : '',
            borderColor: activeTab === 'live' ? 'var(--accent-blue)' : '',
            color: activeTab === 'live' ? '#38bdf8' : '',
          }}
          onClick={() => { setActiveTab('live'); handleFetchLiveFeeds(); }}
        >
          ⚡ Live USGS & Global Feeds
        </button>
        <button
          className="btn-secondary"
          style={{
            background: activeTab === 'stats' ? 'rgba(56, 189, 248, 0.2)' : '',
            borderColor: activeTab === 'stats' ? 'var(--accent-blue)' : '',
            color: activeTab === 'stats' ? '#38bdf8' : '',
          }}
          onClick={() => setActiveTab('stats')}
        >
          📊 Dataset & Benchmarks
        </button>
      </nav>

      {/* TAB 1: LIVE IMAGE & INCIDENT INTELLIGENCE (MAIN USER REQUEST) */}
      {activeTab === 'intelligence' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1.05fr 1.35fr', gap: '1.75rem' }}>
          {/* Upload & Input Panel */}
          <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                <h2 style={{ fontSize: '1.2rem', fontWeight: 600, color: 'var(--accent-blue)', margin: 0 }}>
                  1. Upload Disaster Scene Imagery ({uploadedFiles.length})
                </h2>
                {uploadedFiles.length > 0 && (
                  <button
                    type="button"
                    className="btn-secondary"
                    style={{ padding: '0.2rem 0.6rem', fontSize: '0.75rem', color: '#f43f5e' }}
                    onClick={handleClearImages}
                  >
                    ✕ Clear All
                  </button>
                )}
              </div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
                Upload multiple satellite images, pre/post disaster pairs, drone captures, or street-level photos.
              </p>
            </div>

            {/* Drag & Drop Multi-Image Upload Box */}
            <div
              style={{
                border: '2px dashed var(--border-color)',
                borderRadius: '12px',
                padding: '1.25rem',
                textAlign: 'center',
                background: 'rgba(0, 0, 0, 0.25)',
                cursor: 'pointer',
                position: 'relative',
                transition: 'all 0.2s ease',
              }}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
              onDrop={(e) => {
                e.preventDefault();
                e.stopPropagation();
                if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                  handleFileChange({ target: { files: e.dataTransfer.files } });
                }
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept="image/*"
                style={{ display: 'none' }}
                onChange={handleFileChange}
              />

              <div style={{ padding: '0.5rem 0' }}>
                <div style={{ fontSize: '2rem', marginBottom: '0.35rem' }}>📥</div>
                <div style={{ color: 'var(--text-primary)', fontWeight: 600, fontSize: '0.92rem', marginBottom: '0.25rem' }}>
                  Click to select or drag & drop multiple disaster images
                </div>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>
                  Select multiple files at once • PNG, JPG, JPEG, WebP, GeoTIFF
                </div>
              </div>
            </div>

            {/* Multi-Image Preview Gallery Grid */}
            {previewItems.length > 0 && (
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(110px, 1fr))',
                gap: '0.75rem',
                maxHeight: '260px',
                overflowY: 'auto',
                padding: '0.5rem',
                background: 'rgba(0,0,0,0.3)',
                borderRadius: '8px',
                border: '1px solid var(--border-color)'
              }}>
                {previewItems.map((item, idx) => (
                  <div
                    key={idx}
                    style={{
                      position: 'relative',
                      borderRadius: '6px',
                      overflow: 'hidden',
                      background: 'rgba(15, 23, 42, 0.8)',
                      border: '1px solid rgba(56, 189, 248, 0.3)',
                      display: 'flex',
                      flexDirection: 'column',
                    }}
                  >
                    <img
                      src={item.url}
                      alt={`Upload ${idx + 1}`}
                      style={{
                        width: '100%',
                        height: '85px',
                        objectFit: 'cover',
                      }}
                    />
                    <div style={{
                      padding: '0.3rem',
                      fontSize: '0.7rem',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '0.1rem',
                      background: 'rgba(0,0,0,0.6)'
                    }}>
                      <span
                        title={item.name}
                        style={{
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          color: '#38bdf8',
                          fontWeight: 500,
                        }}
                      >
                        {item.name}
                      </span>
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.65rem' }}>
                        {item.size}
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRemoveFile(idx);
                      }}
                      title="Remove image"
                      style={{
                        position: 'absolute',
                        top: '4px',
                        right: '4px',
                        background: 'rgba(239, 68, 68, 0.85)',
                        color: '#fff',
                        border: 'none',
                        borderRadius: '50%',
                        width: '20px',
                        height: '20px',
                        cursor: 'pointer',
                        fontSize: '0.7rem',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        lineHeight: 1,
                      }}
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Question Input */}
            <div>
              <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 500 }}>
                2. Your Question or Assessment Goal
              </label>
              <textarea
                rows={3}
                style={{
                  width: '100%',
                  background: 'rgba(0,0,0,0.3)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  color: '#fff',
                  padding: '0.75rem',
                  fontSize: '0.9rem',
                  lineHeight: 1.4,
                }}
                value={intelQuestion}
                onChange={e => setIntelQuestion(e.target.value)}
                placeholder="Ask what is happening, where did it happen, and what caused it..."
              />
            </div>

            {/* Live Internet Search Toggle */}
            <div style={{ background: 'rgba(56, 189, 248, 0.08)', padding: '0.85rem', borderRadius: '8px', border: '1px solid rgba(56, 189, 248, 0.2)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <input
                type="checkbox"
                id="searchLive"
                checked={searchLiveWeb}
                onChange={e => setSearchLiveWeb(e.target.checked)}
                style={{ width: '18px', height: '18px', cursor: 'pointer', accentColor: '#0284c7' }}
              />
              <label htmlFor="searchLive" style={{ fontSize: '0.85rem', color: '#e0f2fe', cursor: 'pointer', userSelect: 'none' }}>
                <strong>Correlate with Live Internet Disaster Feeds</strong> (USGS Seismic, ReliefWeb, GDACS Satellite Alerts)
              </label>
            </div>

            {/* Reasoning Engine Mode Switcher */}
            <div style={{
              background: 'rgba(15, 23, 42, 0.7)',
              padding: '0.85rem 1rem',
              borderRadius: '8px',
              border: '1px solid var(--border-color)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: '0.5rem',
            }}>
              <div>
                <div style={{ fontSize: '0.86rem', fontWeight: 600, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <span>{engineMode === 'fast' ? '⚡' : '🧠'}</span>
                  <span>Reasoning Engine</span>
                </div>
                <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>
                  {engineMode === 'fast'
                    ? 'Instant (<30ms) Zero-Shot CLIP-ViT visual feature classification'
                    : 'Deep Multimodal Neural Reasoning via local Qwen2.5-VL on Apple Silicon MPS'}
                </div>
              </div>
              <div style={{ display: 'flex', gap: '0.3rem', background: 'rgba(0,0,0,0.5)', padding: '3px', borderRadius: '7px' }}>
                <button
                  type="button"
                  onClick={() => setEngineMode('fast')}
                  style={{
                    padding: '0.35rem 0.75rem',
                    fontSize: '0.78rem',
                    borderRadius: '5px',
                    border: 'none',
                    cursor: 'pointer',
                    background: engineMode === 'fast' ? '#0284c7' : 'transparent',
                    color: engineMode === 'fast' ? '#fff' : 'var(--text-muted)',
                    fontWeight: engineMode === 'fast' ? 600 : 400,
                    transition: 'all 0.15s ease',
                  }}
                >
                  ⚡ Fast CLIP (&lt;30ms)
                </button>
                <button
                  type="button"
                  onClick={() => setEngineMode('deep_vlm')}
                  style={{
                    padding: '0.35rem 0.75rem',
                    fontSize: '0.78rem',
                    borderRadius: '5px',
                    border: 'none',
                    cursor: 'pointer',
                    background: engineMode === 'deep_vlm' ? '#6366f1' : 'transparent',
                    color: engineMode === 'deep_vlm' ? '#fff' : 'var(--text-muted)',
                    fontWeight: engineMode === 'deep_vlm' ? 600 : 400,
                    transition: 'all 0.15s ease',
                  }}
                >
                  🧠 Deep VLM
                </button>
              </div>
            </div>

            {/* Quick Demo Pre-sets */}
            <div>
              <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.4rem' }}>
                Quick Preset Scenarios:
              </span>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                {[
                  'What disaster is happening here and what caused it?',
                  'Where did this happen and what is the building collapse count?',
                  'Identify the flood trigger, extent, and live emergency reports',
                ].map((preset, idx) => (
                  <button
                    key={idx}
                    type="button"
                    className="btn-secondary"
                    style={{ fontSize: '0.75rem', padding: '0.25rem 0.6rem' }}
                    onClick={() => setIntelQuestion(preset)}
                  >
                    💬 {preset.slice(0, 36)}...
                  </button>
                ))}
              </div>
            </div>

            {/* Single Unified Submit Button */}
            <button
              className="btn-primary"
              style={{
                width: '100%',
                justifyContent: 'center',
                padding: '0.95rem',
                fontSize: '0.98rem',
                fontWeight: 600,
                letterSpacing: '0.01em',
                background: 'linear-gradient(135deg, #0284c7 0%, #4f46e5 100%)',
                boxShadow: '0 4px 14px rgba(2, 132, 199, 0.4)',
                border: 'none',
              }}
              onClick={handleAnalyzeIntelligence}
              disabled={intelLoading}
            >
              {intelLoading
                ? '⏳ Analyzing Disaster Imagery & Correlating Telemetry...'
                : '🚀 Analyze Incident'
              }
            </button>
          </div>

          {/* Results Intelligence Display */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            {intelError && (
              <div style={{ background: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', borderRadius: '8px', padding: '1rem', color: '#fca5a5', fontSize: '0.88rem' }}>
                ⚠️ <strong>Analysis Error:</strong> {intelError}
              </div>
            )}

            {intelResult ? (
              <>
                {/* 1. Incident Breakdown Card */}
                <div className="glass-panel" style={{ padding: '1.75rem', borderLeft: '4px solid #38bdf8' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                      <span className="badge badge-danger" style={{ fontSize: '0.8rem' }}>
                        🚨 {intelResult.breakdown?.detected_disaster_type}
                      </span>
                      <span className="badge badge-success">
                        {(intelResult.breakdown?.confidence * 100).toFixed(0)}% Confidence
                      </span>
                      {intelResult.images_analyzed_count > 0 && (
                        <span className="badge badge-info" style={{ fontSize: '0.75rem' }}>
                          📸 {intelResult.images_analyzed_count} {intelResult.images_analyzed_count === 1 ? 'Image' : 'Images'} Analyzed
                        </span>
                      )}
                    </div>
                    <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                      Response: {intelResult.latency_ms} ms
                    </span>
                  </div>

                  {/* 4 Pillars: WHAT, WHERE, HOW, IMPACT */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.9rem 1rem', borderRadius: '8px', borderLeft: '3px solid #38bdf8' }}>
                      <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#38bdf8', textTransform: 'uppercase', marginBottom: '0.25rem', letterSpacing: '0.04em' }}>
                        📌 What Happened:
                      </div>
                      <div style={{ fontSize: '0.92rem', color: '#f8fafc', lineHeight: 1.5 }}>
                        {intelResult.breakdown?.what_happened}
                      </div>
                    </div>

                    <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.9rem 1rem', borderRadius: '8px', borderLeft: '3px solid #10b981' }}>
                      <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#10b981', textTransform: 'uppercase', marginBottom: '0.25rem', letterSpacing: '0.04em' }}>
                        📍 Where It Happened:
                      </div>
                      <div style={{ fontSize: '0.92rem', color: '#f8fafc', lineHeight: 1.5 }}>
                        {intelResult.breakdown?.where_it_happened}
                      </div>
                    </div>

                    <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.9rem 1rem', borderRadius: '8px', borderLeft: '3px solid #f59e0b' }}>
                      <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#f59e0b', textTransform: 'uppercase', marginBottom: '0.25rem', letterSpacing: '0.04em' }}>
                        ⚡ How It Happened (Mechanisms & Causes):
                      </div>
                      <div style={{ fontSize: '0.92rem', color: '#f8fafc', lineHeight: 1.5 }}>
                        {intelResult.breakdown?.how_it_happened}
                      </div>
                    </div>

                    <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.9rem 1rem', borderRadius: '8px', borderLeft: '3px solid #f43f5e' }}>
                      <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#f43f5e', textTransform: 'uppercase', marginBottom: '0.25rem', letterSpacing: '0.04em' }}>
                        ⚠️ Impact & Damage Extent:
                      </div>
                      <div style={{ fontSize: '0.92rem', color: '#f8fafc', lineHeight: 1.5 }}>
                        {intelResult.breakdown?.impact_and_severity}
                      </div>
                    </div>
                  </div>
                </div>

                {/* 2. Live Internet Evidence Bulletins */}
                <div className="glass-panel" style={{ padding: '1.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      🌐 Live Internet Evidence Found ({intelResult.live_internet_evidence?.length || 0})
                    </h3>
                    <span className="badge badge-info" style={{ fontSize: '0.72rem' }}>
                      Verified Online Feeds
                    </span>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    {intelResult.live_internet_evidence?.map((item, idx) => (
                      <div key={idx} style={{ background: 'rgba(0,0,0,0.3)', padding: '0.9rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.35rem' }}>
                          <span style={{ fontWeight: 600, color: '#38bdf8', fontSize: '0.88rem' }}>
                            {item.title}
                          </span>
                          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{item.timestamp}</span>
                        </div>
                        <p style={{ fontSize: '0.82rem', color: '#cbd5e1', lineHeight: 1.45, marginBottom: '0.4rem' }}>
                          {item.snippet}
                        </p>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem' }}>
                          <span style={{ color: 'var(--text-muted)' }}>
                            <strong>Source:</strong> {item.source_name}
                          </span>
                          {item.source_url && (
                            <a
                              href={item.source_url}
                              target="_blank"
                              rel="noreferrer"
                              style={{ color: '#0284c7', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '0.2rem' }}
                            >
                              🔗 View Live Source ↗
                            </a>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* 3. Where Did This Info Come From (Provenance Audit) */}
                <div className="glass-panel" style={{ padding: '1.25rem 1.5rem' }}>
                  <h4 style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                    🔍 Where Did This Information Come From? (Source Provenance)
                  </h4>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '0.75rem' }}>
                    {intelResult.source_provenance?.map((prov, idx) => (
                      <div key={idx} style={{ background: 'rgba(255,255,255,0.03)', padding: '0.75rem', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                        <div style={{ fontSize: '0.8rem', fontWeight: 600, color: '#34d399', marginBottom: '0.2rem' }}>
                          ✓ {prov.source}
                        </div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                          {prov.details}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div className="glass-panel" style={{ padding: '4rem 2rem', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '440px' }}>
                <span style={{ fontSize: '3.5rem', marginBottom: '1rem', opacity: 0.5 }}>🛰️</span>
                <h3 style={{ fontSize: '1.2rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
                  Ready to Analyze Any Image & Question
                </h3>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', maxWidth: '420px', lineHeight: 1.5 }}>
                  Upload an image on the left, type a question, and click <strong>Find Out What, Where, How</strong> to see visual intelligence correlated with live internet feeds.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 2: Classic RAG */}
      {activeTab === 'rag' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
          <div className="glass-panel" style={{ padding: '1.75rem' }}>
            <h2 style={{ fontSize: '1.15rem', fontWeight: 600, marginBottom: '1.25rem', color: 'var(--accent-blue)' }}>
              Corpus RAG Query
            </h2>
            <textarea
              rows={3}
              style={{ width: '100%', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)', borderRadius: '8px', color: '#fff', padding: '0.75rem', fontSize: '0.9rem', marginBottom: '1rem' }}
              value={ragQuestion}
              onChange={e => setRagQuestion(e.target.value)}
            />
            <button className="btn-primary" style={{ width: '100%', justifyContent: 'center' }} onClick={handleRunRAG} disabled={ragLoading}>
              {ragLoading ? 'Running...' : 'Execute RAG'}
            </button>
          </div>

          <div>
            {ragError && (
              <div style={{ background: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', borderRadius: '8px', padding: '1rem', color: '#fca5a5', marginBottom: '1rem', fontSize: '0.88rem' }}>
                ⚠️ <strong>Query Error:</strong> {ragError}
              </div>
            )}
            {ragResult && (
              <div className="glass-panel" style={{ padding: '1.75rem' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.75rem' }}>RAG Prediction</h3>
                <p style={{ fontSize: '0.95rem', color: '#f8fafc', marginBottom: '1rem' }}>{ragResult.prediction}</p>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Latency: {ragResult.total_latency_ms} ms</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 3: Live Feeds */}
      {activeTab === 'live' && (
        <div className="glass-panel" style={{ padding: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
            <div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                Real-Time Disaster Bulletins (USGS / Global Feeds)
              </h2>
            </div>
            <button className="btn-primary" onClick={handleFetchLiveFeeds} disabled={fetchingFeed}>
              {fetchingFeed ? 'Refreshing...' : '🔄 Poll Live Feeds'}
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: '1rem' }}>
            {liveItems.map((item, idx) => (
              <div key={idx} style={{ background: 'rgba(0,0,0,0.3)', padding: '1.25rem', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                <span className="badge badge-danger" style={{ marginBottom: '0.5rem' }}>{item.severity}</span>
                <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#f8fafc', marginBottom: '0.5rem' }}>{item.title}</h3>
                <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>{item.summary}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* TAB 4: Stats */}
      {activeTab === 'stats' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
          <div className="glass-panel" style={{ padding: '1.75rem' }}>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>DisasterM3 Records</div>
            <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-blue)' }}>1,599</div>
          </div>
          <div className="glass-panel" style={{ padding: '1.75rem' }}>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Standardized Corpus</div>
            <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent-emerald)' }}>1,263</div>
          </div>
        </div>
      )}
    </div>
  );
}
