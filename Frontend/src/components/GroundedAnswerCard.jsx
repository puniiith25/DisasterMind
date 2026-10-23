import React from 'react';
import { ExternalLink, AlertTriangle, Clock, MapPin, CheckCircle, HelpCircle, Satellite } from 'lucide-react';

export default function GroundedAnswerCard({ answer, title = "Evidence-Grounded Intelligence Assessment" }) {
  if (!answer) return null;

  const {
    direct_answer,
    historical_context,
    latest_official_alert,
    supporting_evidence = [],
    source_names_and_urls = [],
    timestamps = {},
    confirmed_location,
    location_provenance,
    location_coords,
    conflicting_evidence,
    missing_information_and_uncertainty,
    practical_next_verification_step,
    recent_imagery,
    sources_checked = [],
  } = answer;

  return (
    <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.25rem', marginTop: '1.5rem' }}>
      {/* Header & Location Provenance */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
            <span style={{ fontSize: '1.2rem' }}>📋</span>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 600, color: 'var(--accent-blue)', margin: 0 }}>
              {title}
            </h3>
          </div>
          {confirmed_location && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              <MapPin size={14} color="#38bdf8" />
              <span>Location: <strong>{confirmed_location}</strong></span>
              <span className="badge" style={{
                background: location_provenance.includes('exif') ? 'rgba(16, 185, 129, 0.15)' : 'rgba(56, 189, 248, 0.15)',
                color: location_provenance.includes('exif') ? '#34d399' : '#38bdf8',
                border: '1px solid rgba(255,255,255,0.1)',
                fontSize: '0.68rem',
              }}>
                Provenance: {location_provenance.replace(/_/g, ' ')}
              </span>
            </div>
          )}
        </div>

        {timestamps?.retrieved_at && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            <Clock size={13} />
            <span>Retrieved: {new Date(timestamps.retrieved_at).toLocaleString()}</span>
          </div>
        )}
      </div>

      {/* 1. Direct Answer */}
      <div style={{
        background: 'rgba(56, 189, 248, 0.08)',
        borderLeft: '4px solid var(--accent-blue)',
        borderRadius: '0 8px 8px 0',
        padding: '1rem 1.25rem',
      }}>
        <div style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--accent-blue)', marginBottom: '0.25rem' }}>
          Direct Grounded Assessment
        </div>
        <p style={{ fontSize: '0.98rem', color: 'var(--text-primary)', lineHeight: 1.6, margin: 0 }}>
          {direct_answer}
        </p>
      </div>

      {/* 2. Latest Official Alert (if any) */}
      {latest_official_alert && (
        <div style={{
          background: 'rgba(244, 63, 94, 0.08)',
          border: '1px solid rgba(244, 63, 94, 0.25)',
          borderRadius: '8px',
          padding: '1rem 1.25rem',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <AlertTriangle size={16} color="#fb7185" />
              <strong style={{ fontSize: '0.85rem', color: '#fb7185', textTransform: 'uppercase' }}>
                Latest Official Report ({latest_official_alert.source})
              </strong>
            </div>
            <span className="badge badge-danger">
              {latest_official_alert.severity}
            </span>
          </div>
          <div style={{ fontSize: '0.92rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
            {latest_official_alert.title}
          </div>
          <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', margin: 0 }}>
            {latest_official_alert.summary}
          </p>
          {latest_official_alert.source_url && (
            <div style={{ marginTop: '0.5rem' }}>
              <a
                href={latest_official_alert.source_url}
                target="_blank"
                rel="noreferrer"
                style={{ fontSize: '0.78rem', color: 'var(--accent-blue)', display: 'inline-flex', alignItems: 'center', gap: '0.25rem', textDecoration: 'none' }}
              >
                Inspect Official Bulletin <ExternalLink size={12} />
              </a>
            </div>
          )}
        </div>
      )}

      {/* 3. Supporting Evidence & Clickable Citations */}
      {supporting_evidence && supporting_evidence.length > 0 && (
        <div>
          <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span>🔗 Supporting Verifiable Evidence ({supporting_evidence.length})</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
            {supporting_evidence.slice(0, 4).map((evi, idx) => (
              <div
                key={evi.id || idx}
                style={{
                  background: 'rgba(0, 0, 0, 0.25)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  padding: '0.75rem 1rem',
                  fontSize: '0.82rem',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span className="badge" style={{
                      background: evi.is_historical ? 'rgba(168, 85, 247, 0.15)' : 'rgba(16, 185, 129, 0.15)',
                      color: evi.is_historical ? '#c084fc' : '#34d399',
                      fontSize: '0.65rem',
                    }}>
                      {evi.is_historical ? 'Historical Baseline' : 'Live Report'}
                    </span>
                    <strong style={{ color: 'var(--text-primary)' }}>{evi.title}</strong>
                  </div>
                  {evi.distance_km != null && (
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      {evi.distance_km} km away
                    </span>
                  )}
                </div>
                <p style={{ color: 'var(--text-secondary)', margin: '0 0 0.35rem 0', lineHeight: 1.4 }}>
                  {evi.snippet}
                </p>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                  <span>Source: {evi.source}</span>
                  {evi.source_url ? (
                    <a
                      href={evi.source_url}
                      target="_blank"
                      rel="noreferrer"
                      style={{ color: 'var(--accent-blue)', display: 'inline-flex', alignItems: 'center', gap: '0.2rem', textDecoration: 'none' }}
                    >
                      View Source <ExternalLink size={11} />
                    </a>
                  ) : (
                    <span>DisasterM3 Bench</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 4. Historical Remote-Sensing Baseline Context */}
      {historical_context && (
        <div style={{
          background: 'rgba(168, 85, 247, 0.08)',
          border: '1px solid rgba(168, 85, 247, 0.2)',
          borderRadius: '8px',
          padding: '0.85rem 1rem',
          fontSize: '0.82rem',
        }}>
          <div style={{ fontWeight: 600, color: '#c084fc', marginBottom: '0.25rem' }}>
            🛰️ Historical Remote-Sensing Baseline Context
          </div>
          <p style={{ color: 'var(--text-secondary)', margin: 0, whiteSpace: 'pre-line', lineHeight: 1.4 }}>
            {historical_context}
          </p>
        </div>
      )}

      {/* 5. Recent Imagery Status & Honest Limitations */}
      {recent_imagery && (
        <div style={{
          background: 'rgba(255, 255, 255, 0.03)',
          border: '1px solid var(--border-color)',
          borderRadius: '8px',
          padding: '0.75rem 1rem',
          fontSize: '0.8rem',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-primary)', fontWeight: 600, marginBottom: '0.25rem' }}>
            <Satellite size={14} color="#38bdf8" />
            <span>Satellite & Aerial Imagery Transparency</span>
            <span className="badge" style={{
              background: recent_imagery.available ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)',
              color: recent_imagery.available ? '#34d399' : '#fbbf24',
              fontSize: '0.65rem',
            }}>
              {recent_imagery.available ? 'Available' : 'Restricted / Low Resolution'}
            </span>
          </div>
          <p style={{ color: 'var(--text-muted)', margin: 0, lineHeight: 1.4 }}>
            {recent_imagery.limitation_reason}
          </p>
        </div>
      )}

      {/* 6. Missing Info, Uncertainty & Next Practical Step */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
        <div style={{
          background: 'rgba(0, 0, 0, 0.2)',
          border: '1px solid var(--border-color)',
          borderRadius: '8px',
          padding: '0.75rem 1rem',
          fontSize: '0.8rem',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: 'var(--accent-amber)', fontWeight: 600, marginBottom: '0.25rem' }}>
            <HelpCircle size={14} />
            <span>Uncertainty & Unverified Conditions</span>
          </div>
          <p style={{ color: 'var(--text-secondary)', margin: 0, lineHeight: 1.4 }}>
            {missing_information_and_uncertainty || "Local structural integrity and municipal street closures are unmonitored."}
          </p>
        </div>

        <div style={{
          background: 'rgba(0, 0, 0, 0.2)',
          border: '1px solid var(--border-color)',
          borderRadius: '8px',
          padding: '0.75rem 1rem',
          fontSize: '0.8rem',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: 'var(--accent-emerald)', fontWeight: 600, marginBottom: '0.25rem' }}>
            <CheckCircle size={14} />
            <span>Next Practical Verification Step</span>
          </div>
          <p style={{ color: 'var(--text-secondary)', margin: 0, lineHeight: 1.4 }}>
            {practical_next_verification_step || "Consult local emergency management or state transportation dispatch."}
          </p>
        </div>
      </div>

      {/* 7. Auditable Sources Checked List */}
      {sources_checked && sources_checked.length > 0 && (
        <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '0.75rem' }}>
          <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginBottom: '0.4rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Authoritative Sources Checked:
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {sources_checked.map((s, idx) => (
              <span
                key={idx}
                style={{
                  background: 'rgba(255, 255, 255, 0.04)',
                  padding: '0.2rem 0.5rem',
                  borderRadius: '4px',
                  fontSize: '0.72rem',
                  color: s.status === 'SUCCESS' ? '#34d399' : s.status === 'NO_RECORDS_FOUND' ? '#94a3b8' : '#fb7185',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.3rem',
                }}
              >
                <span>{s.status === 'SUCCESS' ? '✓' : s.status === 'NO_RECORDS_FOUND' ? '○' : '✕'}</span>
                <strong>{s.provider}</strong>
                <span style={{ color: 'var(--text-muted)' }}>({s.records_count} records, {s.latency_ms}ms)</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
