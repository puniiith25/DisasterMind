import React, { useState } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  MapPin,
  Globe,
  Navigation,
  Copy,
  Check,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  Clock,
  Compass,
  ArrowRight,
} from 'lucide-react';
import DisasterMindEmblem from './DisasterMindEmblem';
import RouteCard from './RouteCard';
import GroundedAnswerCard from './GroundedAnswerCard';

export default function AssistantMessage({
  msg,
  onFollowUp,
  onCalculateRoute,
}) {
  const [copied, setCopied] = useState(false);
  const [customOriginInput, setCustomOriginInput] = useState('');
  const [showOriginPrompt, setShowOriginPrompt] = useState(false);

  const handleCopy = () => {
    const textToCopy = msg.responseData?.summary || msg.responseData?.grounded_answer?.direct_answer || msg.text;
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (msg.isCancelled) {
    return (
      <div className="assistant-msg-wrapper">
        <div className="assistant-avatar">
          <DisasterMindEmblem size={30} />
        </div>
        <div className="assistant-msg-body">
          <div className="status-pill cancelled">
            <span>Investigation paused by user.</span>
          </div>
        </div>
      </div>
    );
  }

  if (msg.isError) {
    return (
      <div className="assistant-msg-wrapper">
        <div className="assistant-avatar">
          <DisasterMindEmblem size={30} />
        </div>
        <div className="assistant-msg-body">
          <div className="status-card error">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontWeight: 700, marginBottom: '0.25rem' }}>
              <AlertTriangle size={16} />
              <span>Investigation Error</span>
            </div>
            <p style={{ margin: 0, fontSize: '0.88rem' }}>{msg.text}</p>
          </div>
        </div>
      </div>
    );
  }

  const res = msg.responseData || {};
  const vlm = res.vlmData || (res.breakdown ? res : null);
  const route = res.routeData || (res.route_geometry ? res : null);
  const locationAns = res.locationData || res.grounded_answer || null;
  const geolocation = vlm?.geolocation_match || res.geolocation_match || null;
  const establishedLoc = res.establishedLocation || geolocation?.place_name || locationAns?.confirmed_location;

  // Primary summary: Answer First!
  const directSummary =
    res.summary ||
    route?.grounded_answer?.direct_answer ||
    route?.safety_advisory ||
    locationAns?.direct_answer ||
    vlm?.summary ||
    msg.text;

  return (
    <div className="assistant-msg-wrapper">
      <div className="assistant-avatar">
        <DisasterMindEmblem size={32} />
      </div>

      <div className="assistant-msg-body">
        {/* Classification & Metadata Badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          {vlm?.breakdown?.detected_disaster_type && (
            <span className="badge badge-orange">
              <ShieldAlert size={13} style={{ marginRight: '4px' }} />
              {vlm.breakdown.detected_disaster_type}
              {vlm.breakdown.confidence && (
                <span> · {Math.round(vlm.breakdown.confidence * 100)}% Match</span>
              )}
            </span>
          )}

          {route && (
            <span className="badge badge-chocolate">
              <Navigation size={12} style={{ marginRight: '4px' }} />
              Corridor: {route.distance_km} km
            </span>
          )}

          {establishedLoc && (
            <span className="badge badge-chocolate">
              <MapPin size={12} style={{ marginRight: '4px' }} />
              {establishedLoc}
            </span>
          )}

          {vlm?.images_analyzed_count > 0 && (
            <span style={{ fontSize: '0.74rem', color: 'var(--color-secondary-text)' }}>
              Analyzed {vlm.images_analyzed_count} scene image{vlm.images_analyzed_count > 1 ? 's' : ''}
            </span>
          )}

          <span style={{ fontSize: '0.72rem', color: 'var(--color-secondary-text)', marginLeft: 'auto' }}>
            {msg.timestamp}
          </span>
        </div>

        {/* 1. Answer First: Grounded Executive Summary */}
        {directSummary && (
          <div className="summary-banner">
            <div style={{ fontSize: '0.72rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--color-burnt-orange)', marginBottom: '0.3rem' }}>
              Evidence-Grounded Intelligence Summary
            </div>
            <p style={{ fontSize: '0.96rem', color: 'var(--color-chocolate)', lineHeight: 1.6, margin: 0, fontWeight: 500 }}>
              {directSummary}
            </p>
          </div>
        )}

        {/* 2. Route Corridor Section (when present) */}
        {route && (
          <RouteCard routeData={route} />
        )}

        {/* 3. Image Analysis Findings (when present) */}
        {vlm?.breakdown && (
          <div
            className="editorial-panel"
            style={{
              padding: '1.25rem 1.4rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.95rem',
              backgroundColor: '#FFFFFF',
            }}
          >
            {/* Observable Damage */}
            <div>
              <div style={{ fontSize: '0.74rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--color-burnt-orange)', fontWeight: 700, marginBottom: '0.25rem' }}>
                Observable Damage & Scene Analysis
              </div>
              <p style={{ fontSize: '0.94rem', color: 'var(--color-chocolate)', lineHeight: 1.6, margin: 0 }}>
                {vlm.breakdown.what_happened}
              </p>
            </div>

            {/* Terrain & Mechanism Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '0.85rem', borderTop: '1px solid var(--color-subtle-border)', paddingTop: '0.85rem' }}>
              <div>
                <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--color-muted-brown)', fontWeight: 700, marginBottom: '0.2rem' }}>
                  Setting & Terrain
                </div>
                <p style={{ fontSize: '0.86rem', color: 'var(--color-secondary-text)', lineHeight: 1.5, margin: 0 }}>
                  {vlm.breakdown.where_it_happened}
                </p>
              </div>

              <div>
                <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--color-muted-brown)', fontWeight: 700, marginBottom: '0.2rem' }}>
                  Physical Mechanism
                </div>
                <p style={{ fontSize: '0.86rem', color: 'var(--color-secondary-text)', lineHeight: 1.5, margin: 0 }}>
                  {vlm.breakdown.how_it_happened}
                </p>
              </div>
            </div>

            {/* Impact & Priority Assessment */}
            {vlm.breakdown.impact_and_severity && (
              <div
                style={{
                  backgroundColor: '#FAF4EC',
                  borderLeft: '3px solid var(--color-burnt-orange)',
                  borderRadius: '0 8px 8px 0',
                  padding: '0.8rem 1rem',
                }}
              >
                <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--color-burnt-orange)', fontWeight: 700, marginBottom: '0.2rem' }}>
                  Impact Severity & Priorities
                </div>
                <p style={{ fontSize: '0.92rem', color: 'var(--color-chocolate)', lineHeight: 1.55, margin: 0 }}>
                  {vlm.breakdown.impact_and_severity}
                </p>
              </div>
            )}
          </div>
        )}

        {/* 4. Likely Location Assessment */}
        {geolocation && (
          <div
            className="editorial-panel"
            style={{
              padding: '1.15rem 1.35rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.75rem',
              backgroundColor: '#FAF4EC',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                <MapPin size={16} color="var(--color-burnt-orange)" />
                <strong style={{ fontSize: '0.92rem', color: 'var(--color-chocolate)', fontFamily: 'var(--font-heading)' }}>
                  Location Assessment: {geolocation.place_name}
                </strong>
              </div>
              <span className="badge badge-orange" style={{ fontSize: '0.68rem' }}>
                Provenance: {String(geolocation.provenance || 'Visual Match').replace(/_/g, ' ')}
              </span>
            </div>

            {geolocation.latitude && geolocation.longitude && (
              <div style={{ fontSize: '0.82rem', color: 'var(--color-chocolate)' }}>
                Coordinates: <strong>{geolocation.latitude.toFixed(4)}, {geolocation.longitude.toFixed(4)}</strong>
              </div>
            )}

            {geolocation.visual_clues && (
              <div style={{ fontSize: '0.82rem', color: 'var(--color-secondary-text)' }}>
                <strong>Observable Clues:</strong> <em>{Array.isArray(geolocation.visual_clues) ? geolocation.visual_clues.join(', ') : geolocation.visual_clues}</em>
              </div>
            )}

            {geolocation.matching_rationale && (
              <p style={{ fontSize: '0.82rem', color: 'var(--color-secondary-text)', margin: 0, lineHeight: 1.45 }}>
                {geolocation.matching_rationale}
              </p>
            )}

            {/* Honest Uncertainty Disclaimer */}
            <div style={{ fontSize: '0.76rem', color: 'var(--color-secondary-text)', lineHeight: 1.45, borderTop: '1px solid var(--color-subtle-border)', paddingTop: '0.5rem' }}>
              ⚠️ <em>Uncertainty disclosure:</em> Candidate location is identified from visible text, architectural patterns, and verified archives. Visual similarity alone does not prove exact coordinates without independent local corroboration.
            </div>

            {/* Contextual Route Actions */}
            {!route && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap', marginTop: '0.35rem' }}>
                <button
                  type="button"
                  onClick={() => onFollowUp(`How far is this location from Bengaluru?`)}
                  className="chip-btn"
                >
                  <Navigation size={12} />
                  <span>Route from Bengaluru</span>
                </button>

                <button
                  type="button"
                  onClick={() => setShowOriginPrompt((prev) => !prev)}
                  className="chip-btn"
                >
                  <Compass size={12} />
                  <span>Route from Custom City...</span>
                </button>

                <button
                  type="button"
                  onClick={() => onFollowUp(`Is this area safe today?`)}
                  className="chip-btn"
                >
                  <ShieldCheck size={12} />
                  <span>Check Current Safety & Alerts</span>
                </button>
              </div>
            )}

            {/* Custom Origin Input Bar */}
            {showOriginPrompt && !route && (
              <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.4rem' }}>
                <input
                  type="text"
                  placeholder="Enter departure city (e.g., Bengaluru, Mumbai, Adana)..."
                  value={customOriginInput}
                  onChange={(e) => setCustomOriginInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && customOriginInput.trim()) {
                      onFollowUp(`What is the route from ${customOriginInput.trim()} to ${geolocation.place_name}?`);
                      setShowOriginPrompt(false);
                    }
                  }}
                  style={{
                    flex: 1,
                    backgroundColor: '#FFFFFF',
                    border: '1px solid var(--color-subtle-border)',
                    borderRadius: '6px',
                    padding: '0.45rem 0.75rem',
                    fontSize: '0.82rem',
                    color: 'var(--color-chocolate)',
                    outline: 'none',
                  }}
                />
                <button
                  type="button"
                  disabled={!customOriginInput.trim()}
                  onClick={() => {
                    if (customOriginInput.trim()) {
                      onFollowUp(`What is the route from ${customOriginInput.trim()} to ${geolocation.place_name}?`);
                      setShowOriginPrompt(false);
                    }
                  }}
                  style={{
                    backgroundColor: 'var(--color-burnt-orange)',
                    color: '#FFFFFF',
                    border: 'none',
                    borderRadius: '6px',
                    padding: '0.45rem 0.85rem',
                    fontSize: '0.8rem',
                    fontWeight: 600,
                    cursor: customOriginInput.trim() ? 'pointer' : 'default',
                    opacity: customOriginInput.trim() ? 1 : 0.6,
                  }}
                >
                  Calculate Route
                </button>
              </div>
            )}
          </div>
        )}

        {/* 5. Live Feed Bulletins (USGS / GDACS) */}
        {vlm?.live_internet_evidence && vlm.live_internet_evidence.length > 0 && (
          <div
            className="editorial-panel"
            style={{
              padding: '1rem 1.25rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.65rem',
              backgroundColor: '#FFFFFF',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.82rem', color: 'var(--color-chocolate)', fontWeight: 700 }}>
              <Globe size={14} color="var(--color-burnt-orange)" />
              <span>Correlated Live Disaster Bulletins ({vlm.live_internet_evidence.length})</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {vlm.live_internet_evidence.map((bulletin, idx) => (
                <div
                  key={idx}
                  style={{
                    backgroundColor: '#FAF4EC',
                    border: '1px solid var(--color-subtle-border)',
                    borderRadius: '8px',
                    padding: '0.75rem 1rem',
                    fontSize: '0.82rem',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                    <strong style={{ color: 'var(--color-chocolate)' }}>{bulletin.title}</strong>
                    {bulletin.source_url && (
                      <a
                        href={bulletin.source_url}
                        target="_blank"
                        rel="noreferrer"
                        style={{ color: 'var(--color-burnt-orange)', display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.72rem', fontWeight: 600, textDecoration: 'none' }}
                      >
                        Inspect Bulletin <ExternalLink size={10} />
                      </a>
                    )}
                  </div>
                  <p style={{ margin: 0, color: 'var(--color-secondary-text)', lineHeight: 1.45 }}>
                    {bulletin.snippet}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 6. Location Grounded Assessment Card */}
        {locationAns && (
          <GroundedAnswerCard
            answer={locationAns}
            title={`Evidence-Grounded Assessment: ${locationAns.confirmed_location || establishedLoc || 'Target Area'}`}
          />
        )}

        {/* 7. Collapsible Technical Details (Preserves Diagnostics cleanly) */}
        <details
          style={{
            backgroundColor: '#FAF4EC',
            border: '1px solid var(--color-subtle-border)',
            borderRadius: '8px',
            padding: '0.65rem 0.95rem',
            fontSize: '0.78rem',
            color: 'var(--color-secondary-text)',
            cursor: 'pointer',
          }}
        >
          <summary style={{ fontWeight: 600, color: 'var(--color-chocolate)', userSelect: 'none' }}>
            Technical & Processing Details (Qwen2.5-VL · Qdrant RRF · Latency Telemetry)
          </summary>
          <div style={{ marginTop: '0.65rem', display: 'flex', flexDirection: 'column', gap: '0.45rem', borderTop: '1px solid var(--color-subtle-border)', paddingTop: '0.65rem', cursor: 'default' }}>
            {res.latency_ms != null && (
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Execution Latency:</span>
                <strong style={{ color: 'var(--color-chocolate)' }}>{Math.round(res.latency_ms)} ms</strong>
              </div>
            )}
            {vlm?.source_provenance && vlm.source_provenance.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', marginTop: '0.2rem' }}>
                <span style={{ fontWeight: 600, color: 'var(--color-chocolate)' }}>Provenance Trail:</span>
                {vlm.source_provenance.map((src, sIdx) => (
                  <span key={sIdx} style={{ fontSize: '0.74rem', color: 'var(--color-secondary-text)' }}>
                    ✓ <strong>{src.source}:</strong> {src.details}
                  </span>
                ))}
              </div>
            )}
          </div>
        </details>

        {/* 8. Action Bar: Copy & Contextual Follow-up Suggestions */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.6rem', marginTop: '0.2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
            <button
              onClick={handleCopy}
              className="action-btn"
              title="Copy answer"
            >
              {copied ? <Check size={13} color="var(--color-success)" /> : <Copy size={13} />}
              <span>{copied ? 'Copied' : 'Copy'}</span>
            </button>

            {/* Contextual Follow-ups */}
            <button
              onClick={() => onFollowUp('What about tomorrow?')}
              className="action-btn"
            >
              <span>What about tomorrow?</span>
            </button>

            {establishedLoc && !route && (
              <button
                onClick={() => onFollowUp(`How far is this location from Bengaluru?`)}
                className="action-btn"
              >
                <span>How far from Bengaluru?</span>
              </button>
            )}

            <button
              onClick={() => onFollowUp('What emergency response equipment and heavy rescue crews should be prioritized for this scene?')}
              className="action-btn"
            >
              <span>Rescue Priorities</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
