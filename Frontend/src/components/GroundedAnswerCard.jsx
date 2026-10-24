import React from 'react';
import { ExternalLink, AlertTriangle, Clock, MapPin, CheckCircle, HelpCircle, Satellite, ShieldCheck, Database } from 'lucide-react';

export default function GroundedAnswerCard({
  answer,
  title = "Evidence-Grounded Intelligence Assessment",
  onSpecifyOrigin,
}) {
  if (!answer) return null;

  const {
    direct_answer,
    historical_context,
    latest_official_alert,
    supporting_evidence = [],
    timestamps = {},
    confirmed_location,
    location_provenance,
    missing_information_and_uncertainty,
    practical_next_verification_step,
    recent_imagery,
    sources_checked = [],
  } = answer;

  return (
    <div
      className="editorial-panel"
      style={{
        padding: '1.25rem 1.4rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '1.15rem',
        backgroundColor: '#FFFCF7',
        border: '1px solid var(--color-subtle-border)',
        borderRadius: '14px',
        boxShadow: 'var(--shadow-card)',
      }}
    >
      {/* Header & Location Provenance */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
            <ShieldCheck size={18} color="var(--color-burnt-orange)" />
            <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--color-chocolate)', margin: 0, fontFamily: 'var(--font-heading)' }}>
              {title}
            </h3>
          </div>
          {confirmed_location && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.84rem', color: 'var(--color-secondary-text)', flexWrap: 'wrap' }}>
              <MapPin size={13} color="var(--color-burnt-orange)" />
              <span>Location: <strong style={{ color: 'var(--color-chocolate)' }}>{confirmed_location}</strong></span>
              <span
                className="badge badge-orange"
                style={{
                  fontSize: '0.68rem',
                  textTransform: 'capitalize',
                }}
              >
                {String(location_provenance || 'Unspecified').replace(/_/g, ' ')}
              </span>
            </div>
          )}
        </div>

        {timestamps?.retrieved_at && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.74rem', color: 'var(--color-secondary-text)' }}>
            <Clock size={12} />
            <span>Telemetry: {new Date(timestamps.retrieved_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
          </div>
        )}
      </div>

      {/* 1. Direct Grounded Answer */}
      {direct_answer && (
        <div
          style={{
            backgroundColor: '#FAF4EC',
            borderLeft: '4px solid var(--color-burnt-orange)',
            borderRadius: '0 8px 8px 0',
            padding: '0.95rem 1.15rem',
          }}
        >
          <div style={{ fontSize: '0.72rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--color-burnt-orange)', marginBottom: '0.25rem' }}>
            Direct Grounded Assessment
          </div>
          <p style={{ fontSize: '0.94rem', color: 'var(--color-chocolate)', lineHeight: 1.6, margin: 0 }}>
            {direct_answer}
          </p>
        </div>
      )}

      {/* 2. Latest Official Alert (USGS / GDACS / NOAA) */}
      {latest_official_alert && (
        <div
          style={{
            backgroundColor: latest_official_alert.severity === 'CRITICAL' || latest_official_alert.severity === 'MAJOR' ? 'var(--color-critical-bg)' : 'var(--color-warning-bg)',
            border: latest_official_alert.severity === 'CRITICAL' || latest_official_alert.severity === 'MAJOR' ? '1px solid var(--color-critical-border)' : '1px solid var(--color-warning-border)',
            borderRadius: '10px',
            padding: '0.95rem 1.15rem',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem', flexWrap: 'wrap', gap: '0.35rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
              <AlertTriangle size={15} color={latest_official_alert.severity === 'CRITICAL' || latest_official_alert.severity === 'MAJOR' ? 'var(--color-critical)' : 'var(--color-warning)'} />
              <strong style={{ fontSize: '0.8rem', color: latest_official_alert.severity === 'CRITICAL' || latest_official_alert.severity === 'MAJOR' ? 'var(--color-critical)' : 'var(--color-warning)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Latest Official Report ({latest_official_alert.source})
              </strong>
            </div>
            <span className={latest_official_alert.severity === 'CRITICAL' || latest_official_alert.severity === 'MAJOR' ? 'badge badge-danger' : 'badge badge-warning'}>
              {latest_official_alert.severity}
            </span>
          </div>
          <div style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--color-chocolate)', marginBottom: '0.25rem' }}>
            {latest_official_alert.title}
          </div>
          <p style={{ fontSize: '0.84rem', color: 'var(--color-chocolate)', margin: 0, lineHeight: 1.5 }}>
            {latest_official_alert.summary}
          </p>
          {latest_official_alert.source_url && (
            <div style={{ marginTop: '0.55rem' }}>
              <a
                href={latest_official_alert.source_url}
                target="_blank"
                rel="noreferrer"
                style={{ fontSize: '0.76rem', color: 'var(--color-burnt-orange)', display: 'inline-flex', alignItems: 'center', gap: '0.3rem', fontWeight: 600, textDecoration: 'none' }}
              >
                Inspect Official Bulletin <ExternalLink size={11} />
              </a>
            </div>
          )}
        </div>
      )}

      {/* 3. Supporting Evidence & Verifiable Citations */}
      {supporting_evidence && supporting_evidence.length > 0 && (
        <div>
          <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--color-chocolate)', marginBottom: '0.55rem', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
            <Database size={14} color="var(--color-burnt-orange)" />
            <span>Supporting Verifiable Evidence ({supporting_evidence.length})</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.55rem' }}>
            {supporting_evidence.slice(0, 4).map((evi, idx) => (
              <div
                key={evi.id || idx}
                style={{
                  backgroundColor: '#FFFFFF',
                  border: '1px solid var(--color-subtle-border)',
                  borderRadius: '8px',
                  padding: '0.8rem 1rem',
                  fontSize: '0.82rem',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem', flexWrap: 'wrap', gap: '0.35rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                    <span className={`badge ${evi.is_historical ? 'badge-chocolate' : 'badge-orange'}`} style={{ fontSize: '0.66rem' }}>
                      {evi.is_historical ? 'Historical Baseline' : 'Live Incident'}
                    </span>
                    <strong style={{ color: 'var(--color-chocolate)', fontSize: '0.86rem' }}>{evi.title}</strong>
                  </div>
                  {evi.distance_km != null && (
                    <span style={{ fontSize: '0.72rem', color: 'var(--color-secondary-text)' }}>
                      {evi.distance_km} km away
                    </span>
                  )}
                </div>
                <p style={{ color: 'var(--color-secondary-text)', margin: '0 0 0.4rem 0', lineHeight: 1.45 }}>
                  {evi.snippet}
                </p>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.72rem', color: 'var(--color-secondary-text)' }}>
                  <span>Source: <strong style={{ color: 'var(--color-chocolate)' }}>{evi.source}</strong></span>
                  {evi.source_url ? (
                    <a
                      href={evi.source_url}
                      target="_blank"
                      rel="noreferrer"
                      style={{ color: 'var(--color-burnt-orange)', display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontWeight: 600, textDecoration: 'none' }}
                    >
                      View Source <ExternalLink size={10} />
                    </a>
                  ) : (
                    <span>DisasterM3 Corpus</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 4. Historical Remote-Sensing Baseline */}
      {historical_context && (
        <div
          style={{
            backgroundColor: '#FAF4EC',
            border: '1px solid var(--color-subtle-border)',
            borderRadius: '8px',
            padding: '0.8rem 1rem',
            fontSize: '0.82rem',
          }}
        >
          <div style={{ fontWeight: 700, color: 'var(--color-muted-brown)', marginBottom: '0.25rem', fontSize: '0.74rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            🛰️ Historical Remote-Sensing Baseline
          </div>
          <p style={{ color: 'var(--color-secondary-text)', margin: 0, whiteSpace: 'pre-line', lineHeight: 1.5 }}>
            {historical_context}
          </p>
        </div>
      )}

      {/* 5. Satellite & Aerial Imagery Transparency */}
      {recent_imagery && (
        <div
          style={{
            backgroundColor: '#FFFFFF',
            border: '1px solid var(--color-subtle-border)',
            borderRadius: '8px',
            padding: '0.8rem 1rem',
            fontSize: '0.8rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', color: 'var(--color-chocolate)', fontWeight: 650, marginBottom: '0.25rem' }}>
            <Satellite size={14} color="var(--color-burnt-orange)" />
            <span>Satellite & Aerial Imagery Transparency</span>
            <span className={`badge ${recent_imagery.available ? 'badge-success' : 'badge-warning'}`} style={{ fontSize: '0.64rem' }}>
              {recent_imagery.available ? 'Pass Available' : 'No Sub-meter Optical Pass'}
            </span>
          </div>
          <p style={{ color: 'var(--color-secondary-text)', margin: 0, lineHeight: 1.45 }}>
            {recent_imagery.limitation_reason}
          </p>
        </div>
      )}

      {/* 6. Uncertainty & Next Practical Verification Step */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '0.75rem' }}>
        <div
          style={{
            backgroundColor: 'var(--color-warning-bg)',
            border: '1px solid var(--color-warning-border)',
            borderRadius: '8px',
            padding: '0.8rem 1rem',
            fontSize: '0.8rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: 'var(--color-warning)', fontWeight: 700, marginBottom: '0.25rem', fontSize: '0.74rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            <HelpCircle size={14} />
            <span>Uncertainty & Coverage Gaps</span>
          </div>
          <p style={{ color: 'var(--color-chocolate)', margin: 0, lineHeight: 1.45 }}>
            {missing_information_and_uncertainty || "Local structural sensor telemetry and municipal street closures are unmonitored in global feeds."}
          </p>
        </div>

        <div
          style={{
            backgroundColor: 'var(--color-success-bg)',
            border: '1px solid var(--color-success-border)',
            borderRadius: '8px',
            padding: '0.8rem 1rem',
            fontSize: '0.8rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: 'var(--color-success)', fontWeight: 700, marginBottom: '0.25rem', fontSize: '0.74rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            <CheckCircle size={14} />
            <span>Next Practical Verification Step</span>
          </div>
          <p style={{ color: 'var(--color-chocolate)', margin: 0, lineHeight: 1.45 }}>
            {practical_next_verification_step || "Consult regional emergency management or state highway patrol dispatch before entering the area."}
          </p>
        </div>
      </div>

      {/* 7. Auditable Sources Checked */}
      {sources_checked && sources_checked.length > 0 && (
        <div style={{ borderTop: '1px solid var(--color-subtle-border)', paddingTop: '0.75rem' }}>
          <div style={{ fontSize: '0.72rem', color: 'var(--color-secondary-text)', marginBottom: '0.35rem', textTransform: 'uppercase', letterSpacing: '0.04em', fontWeight: 600 }}>
            Auditable Sources Queried:
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
            {sources_checked.map((s, idx) => (
              <span
                key={idx}
                style={{
                  backgroundColor: '#FFFFFF',
                  border: '1px solid var(--color-subtle-border)',
                  padding: '0.2rem 0.55rem',
                  borderRadius: '6px',
                  fontSize: '0.72rem',
                  color: s.status === 'SUCCESS' ? 'var(--color-success)' : s.status === 'NO_RECORDS_FOUND' ? 'var(--color-secondary-text)' : 'var(--color-critical)',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.3rem',
                  fontWeight: 500,
                }}
              >
                <span>{s.status === 'SUCCESS' ? '✓' : s.status === 'NO_RECORDS_FOUND' ? '○' : '✕'}</span>
                <strong style={{ color: 'var(--color-chocolate)' }}>{s.provider}</strong>
                <span style={{ color: 'var(--color-secondary-text)' }}>({s.records_count} records)</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
