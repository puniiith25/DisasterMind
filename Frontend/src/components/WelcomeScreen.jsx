import React from 'react';
import {
  Building2,
  Waves,
  Compass,
  Navigation,
  MapPin,
  Flame,
  ArrowRight,
  ShieldCheck,
} from 'lucide-react';
import DisasterMindEmblem from './DisasterMindEmblem';

export default function WelcomeScreen({ onSelectPrompt }) {
  const starterPrompts = [
    {
      title: 'Disaster Scene & Damage Analysis',
      desc: 'Examine concrete shearing, structural collapse, and flood or fire spread.',
      icon: <Building2 size={18} color="var(--color-burnt-orange)" />,
      query: 'Examine this disaster scene: What is the extent of structural damage, collapsed elements, and occupant entrapment risk?',
    },
    {
      title: 'Geographic Location Investigation',
      desc: 'Deduce candidate location using visible text, architecture, and verified archives.',
      icon: <MapPin size={18} color="var(--color-burnt-orange)" />,
      query: 'Where might this disaster scene have been captured? Identify geographical clues and matching archives.',
    },
    {
      title: 'Conditions for a Planned Visit',
      desc: 'Check official disaster bulletins, forecast horizon, and environmental safety.',
      icon: <Compass size={18} color="var(--color-burnt-orange)" />,
      query: 'Can I visit this area next weekend? How is the environment, current disaster alerts, and weather forecast?',
    },
    {
      title: 'Transit Corridor & Route Hazards',
      desc: 'Calculate highway corridor geometry, distance, and road closure hazards.',
      icon: <Navigation size={18} color="var(--color-burnt-orange)" />,
      query: 'What is the highway route, distance, and active road hazard situation from Bengaluru to Wayanad, Kerala?',
    },
  ];

  return (
    <div className="welcome-container animate-fade-in">
      <div className="welcome-header">
        <DisasterMindEmblem size={48} />
        <h1 className="welcome-title">
          DisasterMind Workspace
        </h1>
        <p className="welcome-desc">
          Evidence-grounded multimodal incident analysis, authoritative feed cross-correlation, and transit corridor hazard verification.
        </p>
      </div>

      <div className="starter-card-grid">
        {starterPrompts.map((p, idx) => (
          <div
            key={idx}
            className="starter-prompt-card"
            onClick={() => onSelectPrompt(p.query)}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem' }}>
                {p.icon}
                <strong style={{ fontSize: '0.88rem', color: 'var(--color-chocolate)', fontFamily: 'var(--font-heading)' }}>
                  {p.title}
                </strong>
              </div>
              <ArrowRight size={13} color="var(--color-burnt-orange)" style={{ opacity: 0.7 }} />
            </div>
            <span style={{ fontSize: '0.78rem', color: 'var(--color-secondary-text)', lineHeight: 1.45 }}>
              {p.desc}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
