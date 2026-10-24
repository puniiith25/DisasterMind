import React from 'react';
import { Navigation, AlertTriangle, Clock, ShieldAlert, CheckCircle, ExternalLink, Compass } from 'lucide-react';
import RouteMap from './RouteMap';

export default function RouteCard({ routeData, onExploreAlternative }) {
  if (!routeData) return null;

  const {
    origin,
    destination,
    origin_coords,
    destination_coords,
    distance_km,
    duration_minutes,
    route_found,
    route_geometry,
    detected_hazards = [],
    safety_advisory,
    forward_looking_warning,
    hazard_summary,
    unverified_stretches,
    sources_checked = [],
  } = routeData;

  const hasCritical = detected_hazards.some(
    (h) => h.severity === 'CRITICAL' || h.severity === 'MAJOR'
  );

  return (
    <div
      className="editorial-panel"
      style={{
        padding: '1.25rem 1.4rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem',
        backgroundColor: '#FFFFFF',
        border: '1px solid var(--color-subtle-border)',
        borderRadius: '14px',
        boxShadow: 'var(--shadow-card)',
      }}
    >
      {/* Route Corridor Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', marginBottom: '0.25rem' }}>
            <Navigation size={18} color="var(--color-burnt-orange)" />
            <span style={{ fontSize: '0.74rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--color-burnt-orange)', fontWeight: 700 }}>
              Transit Highway Corridor
            </span>
          </div>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--color-chocolate)', margin: 0, fontFamily: 'var(--font-heading)' }}>
            {origin} <span style={{ color: 'var(--color-burnt-orange)' }}>➔</span> {destination}
          </h3>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', flexWrap: 'wrap' }}>
          {distance_km != null && (
            <span className="badge badge-orange" style={{ fontSize: '0.76rem', padding: '0.25rem 0.65rem' }}>
              <strong>{distance_km} km</strong>
            </span>
          )}
          {duration_minutes != null && (
            <span className="badge badge-chocolate" style={{ fontSize: '0.76rem', padding: '0.25rem 0.65rem' }}>
              <Clock size={12} style={{ marginRight: '4px' }} />
              ~{Math.round(duration_minutes)} min
            </span>
          )}
          <span className={`badge ${route_found ? 'badge-success' : 'badge-danger'}`} style={{ fontSize: '0.74rem' }}>
            {route_found ? 'Direct Highway Computed' : 'No Contiguous Highway'}
          </span>
        </div>
      </div>

      {/* Safety Advisory Banner */}
      {safety_advisory && (
        <div
          style={{
            backgroundColor: hasCritical ? 'var(--color-critical-bg)' : 'var(--color-warning-bg)',
            borderLeft: hasCritical ? '4px solid var(--color-critical)' : '4px solid var(--color-burnt-orange)',
            borderRadius: '0 8px 8px 0',
            padding: '0.85rem 1.15rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.25rem' }}>
            <AlertTriangle size={15} color={hasCritical ? 'var(--color-critical)' : 'var(--color-burnt-orange)'} />
            <strong style={{ fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.04em', color: hasCritical ? 'var(--color-critical)' : 'var(--color-burnt-orange)' }}>
              Corridor Safety Advisory
            </strong>
          </div>
          <p style={{ fontSize: '0.92rem', color: 'var(--color-chocolate)', lineHeight: 1.55, margin: 0 }}>
            {safety_advisory}
          </p>
        </div>
      )}

      {/* Forward-Looking Forecast Warning */}
      {forward_looking_warning && (
        <div
          style={{
            backgroundColor: '#FAF4EC',
            border: '1px solid var(--color-subtle-border)',
            borderRadius: '8px',
            padding: '0.75rem 1rem',
            fontSize: '0.82rem',
            color: 'var(--color-muted-brown)',
            lineHeight: 1.5,
          }}
        >
          <strong>Forecast Horizon Disclosure:</strong> {forward_looking_warning}
        </div>
      )}

      {/* Interactive Leaflet Route Map */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '0.76rem', fontWeight: 700, color: 'var(--color-secondary-text)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Corridor Geometry & Intersecting Hazards
          </span>
          <span style={{ fontSize: '0.72rem', color: 'var(--color-secondary-text)' }}>
            Project OSRM · OpenStreetMap
          </span>
        </div>
        <RouteMap
          routeGeometry={route_geometry}
          hazards={detected_hazards}
          originCoords={origin_coords}
          destinationCoords={destination_coords}
        />
      </div>

      {/* Identified Hazards Along Route */}
      {detected_hazards.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.82rem', fontWeight: 700, color: 'var(--color-chocolate)' }}>
            <ShieldAlert size={15} color="var(--color-burnt-orange)" />
            <span>Identified Alerts Along Corridor ({detected_hazards.length})</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.55rem' }}>
            {detected_hazards.map((h, idx) => {
              const isMajor = h.severity === 'CRITICAL' || h.severity === 'MAJOR';
              return (
                <div
                  key={idx}
                  style={{
                    backgroundColor: '#FAF4EC',
                    border: '1px solid var(--color-subtle-border)',
                    borderRadius: '8px',
                    padding: '0.75rem 1rem',
                    fontSize: '0.82rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.3rem',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.35rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                      <span className={`badge ${isMajor ? 'badge-danger' : 'badge-warning'}`} style={{ fontSize: '0.66rem' }}>
                        {h.severity}
                      </span>
                      <strong style={{ color: 'var(--color-chocolate)' }}>{h.title}</strong>
                    </div>
                    <span style={{ fontSize: '0.72rem', color: 'var(--color-secondary-text)' }}>
                      {h.distance_to_route_km != null ? `${h.distance_to_route_km} km from corridor` : ''}
                    </span>
                  </div>
                  <p style={{ margin: 0, color: 'var(--color-secondary-text)', lineHeight: 1.45 }}>
                    {h.details}
                  </p>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.72rem', color: 'var(--color-secondary-text)', marginTop: '0.15rem' }}>
                    <span>Source: <strong style={{ color: 'var(--color-chocolate)' }}>{h.source}</strong></span>
                    {h.source_url && (
                      <a
                        href={h.source_url}
                        target="_blank"
                        rel="noreferrer"
                        style={{ color: 'var(--color-burnt-orange)', display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontWeight: 600, textDecoration: 'none' }}
                      >
                        Inspect Official Source <ExternalLink size={10} />
                      </a>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Corridor Limitation & Unverified Stretches */}
      {unverified_stretches && (
        <div style={{ fontSize: '0.76rem', color: 'var(--color-secondary-text)', borderTop: '1px solid var(--color-subtle-border)', paddingTop: '0.65rem', lineHeight: 1.45 }}>
          <strong>Telemetry Coverage:</strong> {unverified_stretches}
        </div>
      )}
    </div>
  );
}
