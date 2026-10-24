import React from 'react';

/**
 * Bespoke DisasterMind Emblem
 * Symbolizing geographic contour, location beacon, and evidence pulse.
 */
export default function DisasterMindEmblem({ size = 32, className = '' }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 36 36"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="DisasterMind Emblem"
    >
      {/* Outer rounded container with soft warm cream fill & subtle border */}
      <rect width="36" height="36" rx="9" fill="#F5EBDD" stroke="#E7D9C9" strokeWidth="1" />
      
      {/* Elevation Contours */}
      <path
        d="M6 27C10 24 14 26 18 23C22 20 26 22 30 18"
        stroke="#6E4638"
        strokeWidth="1.2"
        strokeLinecap="round"
        strokeOpacity="0.35"
      />
      <path
        d="M6 31C11 28 15 30 19 27C23 24 27 26 30 22"
        stroke="#6E4638"
        strokeWidth="1.2"
        strokeLinecap="round"
        strokeOpacity="0.2"
      />

      {/* Main Beacon / Location Pin Marker */}
      <path
        d="M18 6C13.58 6 10 9.58 10 14C10 19.4 18 27 18 27C18 27 26 19.4 26 14C26 9.58 22.42 6 18 6Z"
        fill="#38251F"
      />

      {/* Inner Radar Lens Ring */}
      <circle cx="18" cy="14" r="5" fill="#F5EBDD" />

      {/* Primary Burnt Orange Focal Node */}
      <circle cx="18" cy="14" r="3" fill="#C96B3C" />

      {/* Specular Center Point */}
      <circle cx="17.2" cy="13.2" r="1" fill="#FFFFFF" />

      {/* Top Arc Signal */}
      <path
        d="M14 9.5C15.2 8.5 16.5 8 18 8C19.5 8 20.8 8.5 22 9.5"
        stroke="#C96B3C"
        strokeWidth="1.2"
        strokeLinecap="round"
      />
    </svg>
  );
}
