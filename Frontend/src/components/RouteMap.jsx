import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet default icon path issues in bundlers
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

export default function RouteMap({ routeGeometry, hazards = [], originCoords, destinationCoords }) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const layerGroupRef = useRef(null);

  useEffect(() => {
    if (!mapContainerRef.current) return;

    if (!mapInstanceRef.current) {
      const map = L.map(mapContainerRef.current, {
        center: [37.77, -122.41],
        zoom: 7,
        scrollWheelZoom: false,
      });

      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 18,
      }).addTo(map);

      layerGroupRef.current = L.layerGroup().addTo(map);
      mapInstanceRef.current = map;
    }

    const map = mapInstanceRef.current;
    const layers = layerGroupRef.current;
    layers.clearLayers();

    const bounds = [];

    // 1. Draw route polyline in burnt orange
    if (routeGeometry && routeGeometry.coordinates && routeGeometry.coordinates.length > 0) {
      const latLngs = routeGeometry.coordinates.map(pt => [pt[1], pt[0]]);
      L.polyline(latLngs, {
        color: '#C96B3C',
        weight: 5,
        opacity: 0.95,
      }).addTo(layers);

      latLngs.forEach(pt => bounds.push(pt));
    }

    // 2. Add Origin & Destination markers
    if (originCoords && originCoords.latitude && originCoords.longitude) {
      const oMarker = L.circleMarker([originCoords.latitude, originCoords.longitude], {
        radius: 8,
        fillColor: '#286E3B',
        color: '#ffffff',
        weight: 2,
        opacity: 1,
        fillOpacity: 0.95,
      }).addTo(layers);
      oMarker.bindPopup('<strong style="color: #286E3B; font-family: sans-serif;">Origin Departure</strong>');
      bounds.push([originCoords.latitude, originCoords.longitude]);
    }

    if (destinationCoords && destinationCoords.latitude && destinationCoords.longitude) {
      const dMarker = L.circleMarker([destinationCoords.latitude, destinationCoords.longitude], {
        radius: 8,
        fillColor: '#C96B3C',
        color: '#ffffff',
        weight: 2,
        opacity: 1,
        fillOpacity: 0.95,
      }).addTo(layers);
      dMarker.bindPopup('<strong style="color: #C96B3C; font-family: sans-serif;">Destination Target</strong>');
      bounds.push([destinationCoords.latitude, destinationCoords.longitude]);
    }

    // 3. Add Hazard pins
    if (hazards && hazards.length > 0) {
      hazards.forEach(h => {
        if (h.latitude && h.longitude) {
          const isCritical = h.severity === 'CRITICAL' || h.severity === 'MAJOR';
          const pinColor = isCritical ? '#9B2C2C' : '#B45309';

          const hMarker = L.circleMarker([h.latitude, h.longitude], {
            radius: isCritical ? 9 : 7,
            fillColor: pinColor,
            color: '#ffffff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.95,
          }).addTo(layers);

          const popupContent = `
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 12px; line-height: 1.45; color: #38251F;">
              <span style="font-weight: 700; color: ${pinColor};">[${h.severity}] ${h.hazard_type}</span><br/>
              <strong style="color: #38251F;">${h.title}</strong><br/>
              <span style="color: #786A60;">Source: ${h.source}</span><br/>
              <span style="color: #786A60;">Distance to Corridor: ${h.distance_to_route_km} km</span><br/>
              <p style="margin-top: 5px; font-size: 11px; color: #6E4638;">${h.details}</p>
            </div>
          `;
          hMarker.bindPopup(popupContent);
          bounds.push([h.latitude, h.longitude]);
        }
      });
    }

    if (bounds.length > 0) {
      map.fitBounds(bounds, { padding: [40, 40] });
    }

    setTimeout(() => {
      map.invalidateSize();
    }, 200);

  }, [routeGeometry, hazards, originCoords, destinationCoords]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '380px', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--color-subtle-border)', boxShadow: 'var(--shadow-subtle)' }}>
      <div ref={mapContainerRef} style={{ width: '100%', height: '100%' }} />
      <div style={{
        position: 'absolute',
        bottom: '10px',
        right: '10px',
        backgroundColor: 'rgba(255, 252, 247, 0.94)',
        backdropFilter: 'blur(8px)',
        WebkitBackdropFilter: 'blur(8px)',
        border: '1px solid var(--color-subtle-border)',
        padding: '5px 10px',
        borderRadius: '8px',
        fontSize: '11px',
        fontWeight: 500,
        color: '#6E4638',
        boxShadow: 'var(--shadow-subtle)',
        zIndex: 1000,
        pointerEvents: 'none',
      }}>
        🟢 Origin | 🟠 Destination | 🔴 Major Hazard | 🟡 Advisory Alert
      </div>
    </div>
  );
}
