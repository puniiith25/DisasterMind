import React, { useState } from 'react';
import {
  Plus,
  Search,
  MessageSquare,
  Trash2,
  PanelLeftClose,
  Settings,
  X,
  ShieldCheck,
  RefreshCw,
  Globe,
  Database,
  Sliders,
} from 'lucide-react';
import DisasterMindEmblem from './DisasterMindEmblem';

export default function Sidebar({
  sidebarOpen,
  onCloseSidebar,
  sessions = [],
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onClearAllSessions,
  systemStatus = { points: 100, device: 'mps' },
  searchLiveWeb = true,
  onToggleSearchLiveWeb,
}) {
  const [searchQuery, setSearchQuery] = useState('');
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);

  const filteredSessions = sessions.filter((s) => {
    if (!searchQuery.trim()) return true;
    const titleMatch = (s.title || '').toLowerCase().includes(searchQuery.toLowerCase());
    const msgMatch = (s.messages || []).some((m) =>
      (m.text || '').toLowerCase().includes(searchQuery.toLowerCase())
    );
    return titleMatch || msgMatch;
  });

  return (
    <aside className={`workspace-sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
      <div style={{ padding: '1.1rem 1.15rem 0.75rem 1.15rem', display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
        {/* Brand Header & Collapse Toggle */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <DisasterMindEmblem size={30} />
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <span style={{ fontSize: '1.05rem', fontWeight: 800, letterSpacing: '-0.025em', color: 'var(--color-chocolate)', fontFamily: 'var(--font-heading)' }}>
                DisasterMind
              </span>
              <span style={{ fontSize: '0.68rem', color: 'var(--color-muted-brown)', fontWeight: 600, letterSpacing: '0.04em', textTransform: 'uppercase' }}>
                Disaster Intelligence Workspace
              </span>
            </div>
          </div>

          <button
            onClick={onCloseSidebar}
            title="Collapse sidebar"
            className="icon-btn"
          >
            <PanelLeftClose size={18} />
          </button>
        </div>

        {/* New Investigation Button */}
        <button
          onClick={onNewSession}
          className="new-investigation-btn"
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem' }}>
            <Plus size={16} color="var(--color-burnt-orange)" />
            <span>New Investigation</span>
          </div>
          <span className="shortcut-badge">⌘K</span>
        </button>

        {/* Search Conversations Input */}
        <div style={{ position: 'relative' }}>
          <input
            type="text"
            placeholder="Search investigations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{
              width: '100%',
              backgroundColor: '#FFFFFF',
              border: '1px solid var(--color-subtle-border)',
              borderRadius: '8px',
              padding: '0.45rem 0.75rem 0.45rem 2rem',
              fontSize: '0.8rem',
              color: 'var(--color-chocolate)',
              outline: 'none',
              transition: 'border-color 0.15s ease',
            }}
          />
          <Search
            size={13}
            color="var(--color-secondary-text)"
            style={{ position: 'absolute', left: '0.65rem', top: '50%', transform: 'translateY(-50%)' }}
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              style={{
                position: 'absolute',
                right: '0.5rem',
                top: '50%',
                transform: 'translateY(-50%)',
                background: 'none',
                border: 'none',
                color: 'var(--color-secondary-text)',
                cursor: 'pointer',
                padding: '2px',
              }}
            >
              <X size={12} />
            </button>
          )}
        </div>
      </div>

      {/* Session History Section */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '0.25rem 0.85rem', display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
        <div style={{ fontSize: '0.68rem', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--color-secondary-text)', padding: '0.3rem 0.4rem', fontWeight: 700 }}>
          Recent Investigations
        </div>

        {filteredSessions.length === 0 ? (
          <div style={{ padding: '1rem 0.5rem', textAlign: 'center', color: 'var(--color-secondary-text)', fontSize: '0.8rem' }}>
            {searchQuery ? 'No matching investigations' : 'No past conversations yet'}
          </div>
        ) : (
          filteredSessions.map((session) => {
            const isActive = session.id === activeSessionId;
            return (
              <div
                key={session.id}
                onClick={() => onSelectSession(session.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '0.5rem 0.7rem',
                  borderRadius: '8px',
                  backgroundColor: isActive ? '#FFFFFF' : 'transparent',
                  border: isActive ? '1px solid var(--color-burnt-orange)' : '1px solid transparent',
                  color: isActive ? 'var(--color-chocolate)' : 'var(--color-secondary-text)',
                  cursor: 'pointer',
                  fontSize: '0.84rem',
                  boxShadow: isActive ? 'var(--shadow-subtle)' : 'none',
                  transition: 'all 0.15s ease',
                }}
                onMouseEnter={(e) => {
                  if (!isActive) e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.5)';
                }}
                onMouseLeave={(e) => {
                  if (!isActive) e.currentTarget.style.backgroundColor = 'transparent';
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem', overflow: 'hidden' }}>
                  <MessageSquare
                    size={13}
                    style={{
                      flexShrink: 0,
                      opacity: isActive ? 1 : 0.6,
                      color: isActive ? 'var(--color-burnt-orange)' : 'currentColor',
                    }}
                  />
                  <span
                    style={{
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      fontWeight: isActive ? 600 : 400,
                    }}
                  >
                    {session.title || 'New Investigation'}
                  </span>
                </div>

                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteSession(session.id);
                  }}
                  title="Delete session"
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: 'var(--color-secondary-text)',
                    cursor: 'pointer',
                    padding: '2px',
                    borderRadius: '4px',
                    display: 'flex',
                    alignItems: 'center',
                    opacity: isActive ? 0.8 : 0.2,
                    transition: 'all 0.15s ease',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.opacity = '1';
                    e.currentTarget.style.color = 'var(--color-critical)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.opacity = isActive ? '0.8' : '0.2';
                    e.currentTarget.style.color = 'var(--color-secondary-text)';
                  }}
                >
                  <Trash2 size={13} />
                </button>
              </div>
            );
          })
        )}
      </div>

      {/* Bottom Profile / Workspace Telemetry Menu */}
      <div
        style={{
          padding: '0.85rem',
          borderTop: '1px solid var(--color-subtle-border)',
          backgroundColor: '#FAF4EC',
          position: 'relative',
        }}
      >
        {profileMenuOpen && (
          <div
            className="editorial-panel"
            style={{
              position: 'absolute',
              bottom: '100%',
              left: '0.75rem',
              right: '0.75rem',
              marginBottom: '0.5rem',
              padding: '0.95rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.7rem',
              zIndex: 60,
              boxShadow: 'var(--shadow-floating)',
              backgroundColor: '#FFFCF7',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--color-subtle-border)', paddingBottom: '0.45rem' }}>
              <span style={{ fontSize: '0.76rem', fontWeight: 700, color: 'var(--color-chocolate)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Operational Workspace
              </span>
              <button
                onClick={() => setProfileMenuOpen(false)}
                style={{ background: 'none', border: 'none', color: 'var(--color-secondary-text)', cursor: 'pointer' }}
              >
                <X size={14} />
              </button>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--color-secondary-text)' }}>
              <span>Corpus Index</span>
              <span style={{ color: 'var(--color-chocolate)', fontWeight: 650 }}>{systemStatus.points} records</span>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--color-secondary-text)' }}>
              <span>VLM Accelerator</span>
              <span style={{ color: 'var(--color-success)', fontWeight: 650 }}>{systemStatus.device?.toUpperCase() || 'MPS'}</span>
            </div>

            <div
              onClick={onToggleSearchLiveWeb}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '0.4rem 0',
                cursor: 'pointer',
                borderTop: '1px solid var(--color-subtle-border)',
              }}
            >
              <span style={{ fontSize: '0.78rem', color: searchLiveWeb ? 'var(--color-chocolate)' : 'var(--color-secondary-text)' }}>
                Live USGS & GDACS Feeds
              </span>
              <div
                style={{
                  width: '26px',
                  height: '15px',
                  borderRadius: '9999px',
                  backgroundColor: searchLiveWeb ? 'var(--color-burnt-orange)' : '#C4B5A5',
                  position: 'relative',
                  transition: 'background-color 0.2s ease',
                }}
              >
                <div
                  style={{
                    width: '11px',
                    height: '11px',
                    borderRadius: '50%',
                    backgroundColor: '#ffffff',
                    position: 'absolute',
                    top: '2px',
                    left: searchLiveWeb ? '13px' : '2px',
                    transition: 'left 0.2s ease',
                  }}
                />
              </div>
            </div>

            <button
              type="button"
              onClick={() => {
                if (window.confirm('Clear all conversation history?')) {
                  onClearAllSessions();
                  setProfileMenuOpen(false);
                }
              }}
              style={{
                backgroundColor: 'var(--color-critical-bg)',
                border: '1px solid var(--color-critical-border)',
                borderRadius: '6px',
                color: 'var(--color-critical)',
                fontSize: '0.74rem',
                fontWeight: 600,
                padding: '0.4rem',
                cursor: 'pointer',
                textAlign: 'center',
              }}
            >
              Clear All Conversations
            </button>
          </div>
        )}

        {/* Profile Row Button */}
        <div
          onClick={() => setProfileMenuOpen((prev) => !prev)}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0.5rem 0.65rem',
            borderRadius: '8px',
            backgroundColor: '#FFFFFF',
            border: '1px solid var(--color-subtle-border)',
            cursor: 'pointer',
            boxShadow: 'var(--shadow-subtle)',
            transition: 'all 0.15s ease',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <div
              style={{
                width: '28px',
                height: '28px',
                borderRadius: '50%',
                backgroundColor: 'var(--color-chocolate)',
                color: '#FFFFFF',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '0.72rem',
                fontWeight: 700,
              }}
            >
              DA
            </div>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <span style={{ fontSize: '0.82rem', fontWeight: 650, color: 'var(--color-chocolate)' }}>
                Disaster Analyst
              </span>
              <span style={{ fontSize: '0.7rem', color: 'var(--color-secondary-text)' }}>
                Operational Workspace
              </span>
            </div>
          </div>

          <Settings size={14} color="var(--color-secondary-text)" />
        </div>
      </div>
    </aside>
  );
}
