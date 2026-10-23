import React, { useState, useEffect, useRef } from 'react';
import {
  Send,
  Paperclip,
  Image as ImageIcon,
  Plus,
  Trash2,
  Copy,
  Check,
  ExternalLink,
  ChevronDown,
  ChevronRight,
  PanelLeftClose,
  PanelLeft,
  Sparkles,
  ShieldAlert,
  Globe,
  Cpu,
  X,
  Maximize2,
  RefreshCw,
  MessageSquare,
  AlertTriangle,
  Flame,
  Waves,
  Building2,
  Wind,
  Square
} from 'lucide-react';

const STORAGE_KEY = 'disastermind_chat_sessions_v1';

export default function App() {
  // Responsive mobile state
  const [isMobile, setIsMobile] = useState(() => {
    return typeof window !== 'undefined' ? window.innerWidth < 768 : false;
  });

  // Sidebar & System state
  const [sidebarOpen, setSidebarOpen] = useState(() => {
    return typeof window !== 'undefined' ? window.innerWidth >= 768 : true;
  });

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (mobile) {
        setSidebarOpen(false);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const engineMode = 'deep_vlm';
  const [searchLiveWeb, setSearchLiveWeb] = useState(true);
  const [systemStatus, setSystemStatus] = useState({ online: true, points: 1263, vlmReady: true });
  const [selectedPreviewImage, setSelectedPreviewImage] = useState(null);
  const [copiedId, setCopiedId] = useState(null);

  // Chat Sessions
  const [sessions, setSessions] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      }
    } catch (e) {
      console.warn('Failed to parse saved sessions:', e);
    }
    return [
      {
        id: 'session-' + Date.now(),
        title: 'New Disaster Assessment',
        createdAt: new Date().toISOString(),
        image_paths: [],
        messages: [],
      },
    ];
  });

  const [activeSessionId, setActiveSessionId] = useState(() => {
    return sessions[0]?.id || 'session-1';
  });

  // Current session lookup
  const activeSession = sessions.find((s) => s.id === activeSessionId) || sessions[0];

  // Input & attachments state
  const [inputText, setInputText] = useState('');
  const [attachedFiles, setAttachedFiles] = useState([]);
  const [attachedPreviews, setAttachedPreviews] = useState([]);
  const [isSending, setIsSending] = useState(false);

  // Refs
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);
  const messagesEndRef = useRef(null);
  const chatScrollContainerRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Save sessions to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    } catch (e) {
      console.warn('LocalStorage save failed:', e);
    }
  }, [sessions]);

  // Check health and VLM status on mount
  useEffect(() => {
    fetch('/health')
      .then((r) => r.json())
      .then((d) => {
        setSystemStatus((prev) => ({
          ...prev,
          online: d.status === 'HEALTHY',
          points: d.qdrant_points || 1263,
        }));
      })
      .catch(() => {});

    fetch('/api/v1/vlm/status')
      .then((r) => r.json())
      .then((d) => {
        setSystemStatus((prev) => ({
          ...prev,
          vlmReady: d.ready ?? true,
          device: d.device || 'mps',
        }));
      })
      .catch(() => {});
  }, []);

  // Auto-scroll to bottom of chat when new messages appear
  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [activeSession?.messages, isSending]);

  // Adjust textarea height dynamically
  const handleTextareaInput = (e) => {
    setInputText(e.target.value);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 160) + 'px';
    }
  };

  // Create New Chat
  const handleNewChat = () => {
    const newSession = {
      id: 'session-' + Date.now(),
      title: 'New Assessment',
      createdAt: new Date().toISOString(),
      image_paths: [],
      messages: [],
    };
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
    clearInput();
    if (isMobile) setSidebarOpen(false);
  };

  // Delete Chat Session
  const handleDeleteSession = (sessionId, e) => {
    e.stopPropagation();
    if (sessions.length <= 1) {
      // Reset current instead of leaving empty
      const fresh = {
        id: 'session-' + Date.now(),
        title: 'New Assessment',
        createdAt: new Date().toISOString(),
        image_paths: [],
        messages: [],
      };
      setSessions([fresh]);
      setActiveSessionId(fresh.id);
    } else {
      const remaining = sessions.filter((s) => s.id !== sessionId);
      setSessions(remaining);
      if (activeSessionId === sessionId) {
        setActiveSessionId(remaining[0].id);
      }
    }
  };

  // Clear all sessions
  const handleClearAllChats = () => {
    if (window.confirm('Are you sure you want to delete all chat history?')) {
      const fresh = {
        id: 'session-' + Date.now(),
        title: 'New Assessment',
        createdAt: new Date().toISOString(),
        image_paths: [],
        messages: [],
      };
      setSessions([fresh]);
      setActiveSessionId(fresh.id);
      localStorage.removeItem(STORAGE_KEY);
    }
  };

  // Attach images
  const handleFileChange = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      setAttachedFiles((prev) => [...prev, ...files]);
      const previews = files.map((f) => ({
        name: f.name,
        size: (f.size / (1024 * 1024)).toFixed(2) + ' MB',
        url: URL.createObjectURL(f),
      }));
      setAttachedPreviews((prev) => [...prev, ...previews]);
    }
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleRemoveAttached = (index) => {
    setAttachedFiles((prev) => prev.filter((_, i) => i !== index));
    setAttachedPreviews((prev) => {
      URL.revokeObjectURL(prev[index]?.url);
      return prev.filter((_, i) => i !== index);
    });
  };

  const clearInput = () => {
    setInputText('');
    attachedPreviews.forEach((p) => URL.revokeObjectURL(p.url));
    setAttachedFiles([]);
    setAttachedPreviews([]);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Stop / Cancel ongoing generation
  const handleCancelGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsSending(false);
  };

  // Drag and drop images into chat
  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const imageFiles = Array.from(e.dataTransfer.files).filter((f) =>
        f.type.startsWith('image/')
      );
      if (imageFiles.length > 0) {
        setAttachedFiles((prev) => [...prev, ...imageFiles]);
        const previews = imageFiles.map((f) => ({
          name: f.name,
          size: (f.size / (1024 * 1024)).toFixed(2) + ' MB',
          url: URL.createObjectURL(f),
        }));
        setAttachedPreviews((prev) => [...prev, ...previews]);
      }
    }
  };

  // Copy response
  const handleCopy = (id, text) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Send Message
  const handleSendMessage = async (customPrompt) => {
    const questionText = (customPrompt || inputText).trim();
    if (!questionText && attachedFiles.length === 0) return;

    const userMessageId = 'msg-' + Date.now();
    const assistantMessageId = 'msg-' + (Date.now() + 1);

    // Save attached images preview for this user message
    const messageImages = attachedPreviews.map((p) => ({
      name: p.name,
      url: p.url,
    }));

    const newUserMessage = {
      id: userMessageId,
      role: 'user',
      text: questionText || 'Analyze the uploaded disaster imagery.',
      images: messageImages,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    // Update active session with user message and placeholder title
    setSessions((prevSessions) =>
      prevSessions.map((session) => {
        if (session.id === activeSession.id) {
          const updatedMessages = [...session.messages, newUserMessage];
          const newTitle =
            session.messages.length === 0
              ? questionText.slice(0, 38) || 'Disaster Imagery Analysis'
              : session.title;
          return {
            ...session,
            title: newTitle,
            messages: updatedMessages,
          };
        }
        return session;
      })
    );

    // Clear input box
    const filesToSend = [...attachedFiles];
    setInputText('');
    setAttachedFiles([]);
    setAttachedPreviews([]);
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    const controller = new AbortController();
    abortControllerRef.current = controller;
    setIsSending(true);

    try {
      const formData = new FormData();
      formData.append('question', questionText || 'Analyze the uploaded disaster imagery.');
      formData.append('search_live', searchLiveWeb);
      formData.append('engine_mode', engineMode);

      // Multi-turn conversation context: Send recent turns so model reasons based on above conversation
      const priorHistory = activeSession.messages
        .slice(-6)
        .map((m) => {
          if (m.role === 'user') return `User: ${m.text}`;
          const b = m.responseData?.breakdown;
          const ans = b ? `${b.what_happened} ${b.impact_and_severity}` : m.text;
          return `DisasterMind: ${ans}`;
        })
        .join('\n');

      if (priorHistory) {
        formData.append('conversation_history', priorHistory);
      }

      // If user uploaded new files in this turn
      if (filesToSend.length > 0) {
        filesToSend.forEach((file) => {
          formData.append('images', file);
        });
      } else if (activeSession.image_paths && activeSession.image_paths.length > 0) {
        // Reuse existing image paths from this conversation
        formData.append('existing_image_paths', JSON.stringify(activeSession.image_paths));
      }

      const res = await fetch('/api/v1/intelligence/analyze-upload', {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned error (${res.status})`);
      }

      const data = await res.json();

      const newAssistantMessage = {
        id: assistantMessageId,
        role: 'assistant',
        text: data.summary,
        responseData: data,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      // Update session with assistant response and store image paths for follow-up turns
      setSessions((prevSessions) =>
        prevSessions.map((session) => {
          if (session.id === activeSession.id) {
            const combinedPaths = Array.from(
              new Set([...(session.image_paths || []), ...(data.image_paths || [])])
            );
            return {
              ...session,
              image_paths: combinedPaths,
              messages: [...session.messages, newAssistantMessage],
            };
          }
          return session;
        })
      );
    } catch (err) {
      if (err.name === 'AbortError' || controller.signal.aborted) {
        const cancelledMessage = {
          id: assistantMessageId,
          role: 'assistant',
          isCancelled: true,
          text: 'Assessment stopped by user.',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        };

        setSessions((prevSessions) =>
          prevSessions.map((session) => {
            if (session.id === activeSession.id) {
              return {
                ...session,
                messages: [...session.messages, cancelledMessage],
              };
            }
            return session;
          })
        );
        return;
      }

      const errorMessage = {
        id: assistantMessageId,
        role: 'assistant',
        isError: true,
        text: `Analysis failed: ${err.message}. Please check that the DisasterMind backend server is running.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setSessions((prevSessions) =>
        prevSessions.map((session) => {
          if (session.id === activeSession.id) {
            return {
              ...session,
              messages: [...session.messages, errorMessage],
            };
          }
          return session;
        })
      );
    } finally {
      setIsSending(false);
      abortControllerRef.current = null;
    }
  };

  // Keyboard shortcut: Enter sends message (Shift+Enter inserts newline), Escape cancels
  const handleKeyDown = (e) => {
    if (e.key === 'Escape' && isSending) {
      e.preventDefault();
      handleCancelGeneration();
      return;
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isSending) {
        handleSendMessage();
      }
    }
  };

  // Starter Prompts
  const starterPrompts = [
    {
      title: 'Assess Building Damage',
      desc: 'Evaluate collapsed concrete elements, shear cracks, and rubble stability',
      icon: <Building2 size={20} className="text-amber-400" />,
      query: 'Examine this disaster scene: What is the extent of structural damage, collapsed elements, and occupant entrapment risk?',
    },
    {
      title: 'Flood Inundation & Rescue',
      desc: 'Measure submerged depth, road cutoffs, and utility vulnerability',
      icon: <Waves size={20} className="text-cyan-400" />,
      query: 'Assess this flood scene: What is the estimated water depth, submerged road status, and immediate evacuation priorities?',
    },
    {
      title: 'Wildfire Spread & Hazards',
      desc: 'Analyze active flame front, smoke direction, and perimeter risk',
      icon: <Flame size={20} className="text-rose-400" />,
      query: 'Analyze this wildfire imagery: What is the smoke plume density, flame propagation risk, and containment urgency?',
    },
    {
      title: 'Storm & Hurricane Debris',
      desc: 'Analyze roof peeling, downed power lines, and tree blockages',
      icon: <Wind size={20} className="text-sky-400" />,
      query: 'What cyclonic storm or hurricane damage has occurred here? List damaged infrastructure and road blockages.',
    },
  ];

  return (
    <div className="chat-container" onDragOver={handleDragOver} onDrop={handleDrop}>
      {/* Mobile backdrop */}
      {isMobile && sidebarOpen && (
        <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />
      )}

      {/* ============================================================ */}
      {/* SIDEBAR (ChatGPT Style)                                      */}
      {/* ============================================================ */}
      <aside className={`chat-sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
        <div style={{ padding: '0.85rem 0.85rem 0.5rem 0.85rem', display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
          {/* Top Brand & Collapse */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.2rem 0.35rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <img
                src="/logo-icon.png"
                alt="DisasterMind Icon"
                style={{
                  width: '26px',
                  height: '26px',
                  objectFit: 'contain',
                  filter: 'drop-shadow(0 2px 6px rgba(0, 114, 194, 0.4))',
                }}
              />
              <img
                src="/logo-white.png"
                alt="DisasterMind Logo"
                style={{
                  height: '22px',
                  width: 'auto',
                  objectFit: 'contain',
                }}
              />
            </div>

            <button
              onClick={() => setSidebarOpen(false)}
              title="Close sidebar"
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--text-muted)',
                cursor: 'pointer',
                padding: '4px',
                borderRadius: '6px',
                display: 'flex',
                alignItems: 'center',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.color = '#fff')}
              onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-muted)')}
            >
              <PanelLeftClose size={18} />
            </button>
          </div>

          {/* New Chat Button */}
          <button
            onClick={handleNewChat}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              width: '100%',
              backgroundColor: 'var(--bg-card)',
              color: 'var(--text-main)',
              border: '1px solid var(--border-light)',
              borderRadius: '10px',
              padding: '0.65rem 0.85rem',
              fontSize: '0.875rem',
              fontWeight: 500,
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = 'var(--accent-sky)';
              e.currentTarget.style.backgroundColor = 'var(--bg-card-hover)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'var(--border-light)';
              e.currentTarget.style.backgroundColor = 'var(--bg-card)';
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem' }}>
              <Plus size={16} />
              <span>New chat</span>
            </div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', border: '1px solid var(--border-light)', padding: '1px 5px', borderRadius: '4px' }}>
              Ctrl+K
            </span>
          </button>
        </div>

        {/* Sessions History List */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '0.5rem 0.65rem', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', padding: '0.4rem 0.5rem', fontWeight: 600 }}>
            Recent Conversations
          </div>

          {sessions.map((session) => {
            const isActive = session.id === activeSession.id;
            return (
              <div
                key={session.id}
                onClick={() => {
                  setActiveSessionId(session.id);
                  clearInput();
                  if (isMobile) setSidebarOpen(false);
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '0.55rem 0.75rem',
                  borderRadius: '8px',
                  backgroundColor: isActive ? 'var(--bg-session-active)' : 'transparent',
                  border: isActive ? '1px solid var(--border-sky)' : '1px solid transparent',
                  color: isActive ? '#ffffff' : 'var(--text-secondary)',
                  cursor: 'pointer',
                  fontSize: '0.85rem',
                  transition: 'all 0.15s ease',
                  group: 'session-row',
                }}
                onMouseEnter={(e) => {
                  if (!isActive) e.currentTarget.style.backgroundColor = 'var(--bg-card-hover)';
                }}
                onMouseLeave={(e) => {
                  if (!isActive) e.currentTarget.style.backgroundColor = 'transparent';
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem', overflow: 'hidden' }}>
                  <MessageSquare size={14} style={{ flexShrink: 0, opacity: isActive ? 1 : 0.6 }} />
                  <span
                    style={{
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      fontWeight: isActive ? 500 : 400,
                    }}
                  >
                    {session.title || 'New Assessment'}
                  </span>
                </div>

                <button
                  onClick={(e) => handleDeleteSession(session.id, e)}
                  title="Delete chat"
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: 'var(--text-muted)',
                    cursor: 'pointer',
                    padding: '2px',
                    borderRadius: '4px',
                    display: 'flex',
                    alignItems: 'center',
                    opacity: isActive ? 0.7 : 0.3,
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.opacity = '1';
                    e.currentTarget.style.color = '#f87171';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.opacity = isActive ? '0.7' : '0.3';
                    e.currentTarget.style.color = 'var(--text-muted)';
                  }}
                >
                  <Trash2 size={13} />
                </button>
              </div>
            );
          })}
        </div>

        {/* Sidebar Controls Footer */}
        <div
          style={{
            padding: '0.85rem',
            borderTop: '1px solid var(--border-light)',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.75rem',
            backgroundColor: '#141414',
          }}
        >
          {/* Reasoning Engine Indicator */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0.45rem 0.55rem',
              borderRadius: '8px',
              backgroundColor: 'var(--bg-card)',
              border: '1px solid var(--border-light)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.75rem' }}>
              <Cpu size={13} color="var(--accent-sky)" />
              <span style={{ color: 'var(--text-secondary)' }}>Reasoning Engine</span>
            </div>
            <span style={{ fontSize: '0.75rem', color: 'var(--accent-sky)', fontWeight: 600 }}>
              Deep Neural
            </span>
          </div>

          {/* Live Web Telemetry Toggle */}
          <div
            onClick={() => setSearchLiveWeb((prev) => !prev)}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0.45rem 0.55rem',
              borderRadius: '8px',
              backgroundColor: 'var(--bg-card)',
              border: '1px solid var(--border-light)',
              cursor: 'pointer',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.75rem' }}>
              <Globe size={13} color={searchLiveWeb ? 'var(--accent-sky)' : 'var(--text-muted)'} />
              <span style={{ color: searchLiveWeb ? '#ececec' : 'var(--text-muted)' }}>USGS Live Telemetry</span>
            </div>
            <div
              style={{
                width: '28px',
                height: '16px',
                borderRadius: '9999px',
                backgroundColor: searchLiveWeb ? 'var(--accent-sky)' : '#334155',
                position: 'relative',
                transition: 'background-color 0.2s ease',
              }}
            >
              <div
                style={{
                  width: '12px',
                  height: '12px',
                  borderRadius: '50%',
                  backgroundColor: '#ffffff',
                  position: 'absolute',
                  top: '2px',
                  left: searchLiveWeb ? '14px' : '2px',
                  transition: 'left 0.2s ease',
                }}
              />
            </div>
          </div>

          {/* System Online Badge & Clear history */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingTop: '0.2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              <span className="online-dot" />
              <span>Indexed: {systemStatus.points} pts</span>
            </div>

            <button
              onClick={handleClearAllChats}
              title="Clear all chat history"
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--text-muted)',
                cursor: 'pointer',
                fontSize: '0.72rem',
                textDecoration: 'underline',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.color = '#f87171')}
              onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-muted)')}
            >
              Clear all
            </button>
          </div>
        </div>
      </aside>

      {/* ============================================================ */}
      {/* MAIN CHAT VIEWPORT                                           */}
      {/* ============================================================ */}
      <main className="chat-main">
        {/* Top Header Bar */}
        <header
          className="chat-header"
          style={{
            height: '52px',
            borderBottom: '1px solid var(--border-light)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 1rem',
            backgroundColor: 'var(--bg-header)',
            backdropFilter: 'blur(12px)',
            zIndex: 20,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            {(!sidebarOpen || isMobile) && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                <button
                  onClick={() => setSidebarOpen((prev) => !prev)}
                  title={sidebarOpen ? "Close sidebar" : "Open sidebar"}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: 'var(--text-secondary)',
                    cursor: 'pointer',
                    padding: '6px',
                    borderRadius: '6px',
                    display: 'flex',
                    alignItems: 'center',
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.color = '#fff')}
                  onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-secondary)')}
                >
                  <PanelLeft size={19} />
                </button>
                <img
                  src="/logo-icon.png"
                  alt="DisasterMind Logo"
                  style={{
                    width: '24px',
                    height: '24px',
                    objectFit: 'contain',
                    filter: 'drop-shadow(0 2px 6px rgba(0, 114, 194, 0.4))',
                  }}
                />
              </div>
            )}

            {/* Model Badge */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.45rem',
                padding: '0.25rem 0.65rem',
                borderRadius: '8px',
                backgroundColor: 'rgba(56, 189, 248, 0.08)',
                border: '1px solid var(--border-light)',
                fontSize: '0.85rem',
                fontWeight: 500,
                color: '#f0f6fc',
              }}
            >
              <img
                src="/logo-icon.png"
                alt="DisasterMind"
                style={{ width: '15px', height: '15px', objectFit: 'contain' }}
              />
              <span>DisasterMind 3B</span>
              <span className="hide-on-mobile" style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                · Deep Neural VLM
              </span>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {activeSession?.image_paths?.length > 0 && (
              <span
                style={{
                  fontSize: '0.75rem',
                  padding: '0.2rem 0.55rem',
                  borderRadius: '9999px',
                  backgroundColor: 'rgba(16, 163, 127, 0.15)',
                  color: '#34d399',
                  border: '1px solid rgba(16, 163, 127, 0.3)',
                }}
              >
                📸 Image Context Active ({activeSession.image_paths.length})
              </span>
            )}

            <button
              onClick={handleNewChat}
              style={{
                background: 'transparent',
                border: '1px solid var(--border-light)',
                borderRadius: '8px',
                color: 'var(--text-secondary)',
                padding: '0.35rem 0.65rem',
                fontSize: '0.8rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'var(--border-hover)';
                e.currentTarget.style.color = '#fff';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'var(--border-light)';
                e.currentTarget.style.color = 'var(--text-secondary)';
              }}
            >
              <Plus size={14} />
              <span>New</span>
            </button>
          </div>
        </header>

        {/* Scrollable Messages Thread */}
        <div className="messages-container" ref={chatScrollContainerRef}>
          {activeSession.messages.length === 0 ? (
            /* Empty State / Welcome Screen */
            <div
              style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '2rem 1.5rem 10rem 1.5rem',
                maxWidth: '780px',
                margin: '0 auto',
                width: '100%',
                textAlign: 'center',
              }}
            >
              <img
                src="/logo-white.png"
                alt="DisasterMind: AI For A Safer Tomorrow"
                className="hero-logo"
                style={{
                  height: '68px',
                  width: 'auto',
                  maxWidth: '320px',
                  objectFit: 'contain',
                  marginBottom: '1.25rem',
                  filter: 'drop-shadow(0 6px 20px rgba(0, 114, 194, 0.25))',
                }}
              />

              <h1
                className="hero-title"
                style={{
                  fontSize: '1.85rem',
                  fontWeight: 600,
                  color: '#ffffff',
                  marginBottom: '0.5rem',
                  letterSpacing: '-0.03em',
                }}
              >
                How can I assist your disaster assessment?
              </h1>
              <p
                className="hero-desc"
                style={{
                  color: 'var(--text-secondary)',
                  fontSize: '0.95rem',
                  maxWidth: '540px',
                  lineHeight: 1.6,
                  marginBottom: '2.5rem',
                }}
              >
                Upload any disaster photo, aerial capture, or question. DisasterMind assesses structural integrity,
                identifies damage causes, links live USGS feeds, and remembers context across follow-ups.
              </p>

              {/* Starter Quick Actions Grid */}
              <div
                className="starter-card-grid"
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
                  gap: '0.75rem',
                  width: '100%',
                }}
              >
                {starterPrompts.map((starter, i) => (
                  <div
                    key={i}
                    className="starter-card"
                    onClick={() => handleSendMessage(starter.query)}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem' }}>
                      {starter.icon}
                      <span style={{ fontWeight: 600, fontSize: '0.88rem', color: '#f3f4f6' }}>
                        {starter.title}
                      </span>
                    </div>
                    <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                      {starter.desc}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            /* Active Messages List */
            <div className="messages-inner">
              {activeSession.messages.map((msg) => (
                <div
                  key={msg.id}
                  className="animate-fade-in"
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.5rem',
                    alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start',
                    width: '100%',
                  }}
                >
                  {/* USER MESSAGE */}
                  {msg.role === 'user' ? (
                    <div
                      style={{
                        maxWidth: '85%',
                        backgroundColor: 'var(--bg-user-bubble)',
                        border: '1px solid var(--border-light)',
                        borderRadius: '20px 20px 4px 20px',
                        padding: '0.85rem 1.15rem',
                        color: 'var(--text-main)',
                        boxShadow: '0 2px 10px rgba(0,0,0,0.2)',
                      }}
                    >
                      {/* Uploaded image previews */}
                      {msg.images && msg.images.length > 0 && (
                        <div
                          style={{
                            display: 'flex',
                            flexWrap: 'wrap',
                            gap: '0.5rem',
                            marginBottom: '0.65rem',
                          }}
                        >
                          {msg.images.map((img, idx) => (
                            <div
                              key={idx}
                              style={{
                                position: 'relative',
                                borderRadius: '10px',
                                overflow: 'hidden',
                                border: '1px solid rgba(255,255,255,0.15)',
                                cursor: 'pointer',
                              }}
                              onClick={() => setSelectedPreviewImage(img.url)}
                            >
                              <img
                                src={img.url}
                                alt={img.name}
                                style={{
                                  width: '140px',
                                  height: '100px',
                                  objectFit: 'cover',
                                  display: 'block',
                                }}
                              />
                              <div
                                style={{
                                  position: 'absolute',
                                  bottom: 0,
                                  left: 0,
                                  right: 0,
                                  padding: '2px 6px',
                                  fontSize: '0.65rem',
                                  backgroundColor: 'rgba(0,0,0,0.65)',
                                  color: '#fff',
                                  whiteSpace: 'nowrap',
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                }}
                              >
                                {img.name}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}

                      <p style={{ fontSize: '0.95rem', lineHeight: 1.55, whiteSpace: 'pre-wrap' }}>
                        {msg.text}
                      </p>
                      <div
                        style={{
                          fontSize: '0.7rem',
                          color: 'var(--text-muted)',
                          textAlign: 'right',
                          marginTop: '0.35rem',
                        }}
                      >
                        {msg.timestamp}
                      </div>
                    </div>
                  ) : (
                    /* ASSISTANT MESSAGE */
                    <div
                      style={{
                        display: 'flex',
                        gap: '1rem',
                        width: '100%',
                        padding: '0.5rem 0',
                      }}
                    >
                      {/* Avatar */}
                      <div
                        style={{
                          width: '32px',
                          height: '32px',
                          borderRadius: '8px',
                          background: 'rgba(255, 255, 255, 0.05)',
                          border: '1px solid rgba(255, 255, 255, 0.1)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0,
                          boxShadow: '0 2px 8px rgba(0, 114, 194, 0.2)',
                          overflow: 'hidden',
                        }}
                      >
                        <img
                          src="/logo-icon.png"
                          alt="DisasterMind AI"
                          style={{ width: '22px', height: '22px', objectFit: 'contain' }}
                        />
                      </div>

                      {/* Content Box */}
                      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                        {msg.isCancelled ? (
                          <div
                            style={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '0.5rem',
                              backgroundColor: 'rgba(255, 255, 255, 0.04)',
                              border: '1px solid rgba(255, 255, 255, 0.08)',
                              borderRadius: '8px',
                              padding: '0.6rem 0.9rem',
                              color: 'var(--text-muted)',
                              fontSize: '0.86rem',
                              width: 'fit-content',
                            }}
                          >
                            <Square size={13} fill="currentColor" color="#ef4444" />
                            <span>Assessment stopped by user.</span>
                          </div>
                        ) : msg.isError ? (
                          <div
                            style={{
                              backgroundColor: 'rgba(244, 63, 94, 0.1)',
                              border: '1px solid rgba(244, 63, 94, 0.3)',
                              borderRadius: '12px',
                              padding: '1rem',
                              color: '#fb7185',
                              fontSize: '0.9rem',
                            }}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem', fontWeight: 600 }}>
                              <AlertTriangle size={16} />
                              <span>Error Executing Analysis</span>
                            </div>
                            <p>{msg.text}</p>
                          </div>
                        ) : msg.responseData ? (
                          /* Structured Incident Intelligence Card */
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                            {/* Disaster Header Badge & Latency */}
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', flexWrap: 'wrap' }}>
                              <span
                                style={{
                                  display: 'inline-flex',
                                  alignItems: 'center',
                                  gap: '0.4rem',
                                  padding: '0.25rem 0.75rem',
                                  borderRadius: '9999px',
                                  backgroundColor: 'rgba(16, 163, 127, 0.18)',
                                  color: '#34d399',
                                  border: '1px solid rgba(16, 163, 127, 0.35)',
                                  fontSize: '0.8rem',
                                  fontWeight: 600,
                                  textTransform: 'uppercase',
                                  letterSpacing: '0.04em',
                                }}
                              >
                                <ShieldAlert size={14} />
                                {msg.responseData.breakdown?.detected_disaster_type || 'DISASTER INCIDENT'}
                                <span>·</span>
                                {Math.round((msg.responseData.breakdown?.confidence || 0.88) * 100)}% Confidence
                              </span>

                              {msg.responseData.latency_ms > 0 && (
                                <span
                                  style={{
                                    fontSize: '0.72rem',
                                    color: 'var(--text-muted)',
                                    backgroundColor: 'rgba(255, 255, 255, 0.05)',
                                    padding: '0.2rem 0.5rem',
                                    borderRadius: '6px',
                                  }}
                                >
                                  ⚡ {Math.round(msg.responseData.latency_ms)} ms
                                </span>
                              )}
                            </div>

                            {/* 4 Pillars Breakdown Card */}
                            <div
                              style={{
                                backgroundColor: 'var(--bg-card)',
                                border: '1px solid var(--border-light)',
                                borderRadius: '14px',
                                overflow: 'hidden',
                                boxShadow: '0 4px 16px rgba(0, 0, 0, 0.25)',
                              }}
                            >
                              {/* WHAT */}
                              <div style={{ padding: '0.9rem 1.15rem', borderBottom: '1px solid var(--border-light)' }}>
                                <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#38bdf8', fontWeight: 600, marginBottom: '0.25rem' }}>
                                  What Happened (Observable Damage)
                                </div>
                                <div style={{ fontSize: '0.92rem', color: '#f3f4f6', lineHeight: 1.55 }}>
                                  {msg.responseData.breakdown?.what_happened}
                                </div>
                              </div>

                              {/* WHERE & HOW Grid */}
                              <div
                                className="pillar-grid"
                                style={{
                                  display: 'grid',
                                  gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
                                  borderBottom: '1px solid var(--border-light)',
                                }}
                              >
                                <div className="pillar-border-right" style={{ padding: '0.85rem 1.15rem', borderRight: '1px solid var(--border-light)' }}>
                                  <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#fbbf24', fontWeight: 600, marginBottom: '0.25rem' }}>
                                    Where (Setting & Terrain)
                                  </div>
                                  <div style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                                    {msg.responseData.breakdown?.where_it_happened}
                                  </div>
                                </div>

                                <div style={{ padding: '0.85rem 1.15rem' }}>
                                  <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#38bdf8', fontWeight: 600, marginBottom: '0.25rem' }}>
                                    How (Physical Mechanism)
                                  </div>
                                  <div style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                                    {msg.responseData.breakdown?.how_it_happened}
                                  </div>
                                </div>
                              </div>

                              {/* IMPACT & ANSWER */}
                              <div style={{ padding: '0.95rem 1.15rem', backgroundColor: 'rgba(56, 189, 248, 0.06)' }}>
                                <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#38bdf8', fontWeight: 600, marginBottom: '0.25rem' }}>
                                  Impact & Assessment Answer
                                </div>
                                <div style={{ fontSize: '0.92rem', color: '#ffffff', lineHeight: 1.55, fontWeight: 450 }}>
                                  {msg.responseData.breakdown?.impact_and_severity}
                                </div>
                              </div>
                            </div>

                            {/* Live USGS Evidence Bulletins (If any correlated) */}
                            {msg.responseData.live_internet_evidence &&
                              msg.responseData.live_internet_evidence.length > 0 && (
                                <div
                                  style={{
                                    backgroundColor: 'rgba(56, 189, 248, 0.05)',
                                    border: '1px solid rgba(56, 189, 248, 0.2)',
                                    borderRadius: '12px',
                                    padding: '0.85rem 1rem',
                                    display: 'flex',
                                    flexDirection: 'column',
                                    gap: '0.5rem',
                                  }}
                                >
                                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.78rem', color: '#38bdf8', fontWeight: 600 }}>
                                    <Globe size={14} />
                                    <span>Correlated Live Internet Telemetry (USGS)</span>
                                  </div>

                                  {msg.responseData.live_internet_evidence.map((bulletin, bIdx) => (
                                    <div
                                      key={bIdx}
                                      style={{
                                        fontSize: '0.82rem',
                                        color: '#cbd5e1',
                                        backgroundColor: 'rgba(0, 0, 0, 0.25)',
                                        borderRadius: '8px',
                                        padding: '0.65rem 0.85rem',
                                        display: 'flex',
                                        flexDirection: 'column',
                                        gap: '0.25rem',
                                      }}
                                    >
                                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                        <span style={{ fontWeight: 600, color: '#f8fafc' }}>
                                          {bulletin.title}
                                        </span>
                                        {bulletin.source_url && (
                                          <a
                                            href={bulletin.source_url}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            style={{ color: '#38bdf8', display: 'flex', alignItems: 'center', gap: '2px', fontSize: '0.72rem' }}
                                          >
                                            View <ExternalLink size={11} />
                                          </a>
                                        )}
                                      </div>
                                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.78rem' }}>
                                        {bulletin.snippet}
                                      </p>
                                    </div>
                                  ))}
                                </div>
                              )}

                            {/* Source Provenance Tags */}
                            {msg.responseData.source_provenance && (
                              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginTop: '0.2rem' }}>
                                {msg.responseData.source_provenance.map((src, sIdx) => (
                                  <span
                                    key={sIdx}
                                    style={{
                                      fontSize: '0.72rem',
                                      color: 'var(--text-muted)',
                                      backgroundColor: 'rgba(255, 255, 255, 0.04)',
                                      border: '1px solid var(--border-light)',
                                      padding: '0.2rem 0.55rem',
                                      borderRadius: '6px',
                                    }}
                                  >
                                    ✓ {src.source}: {src.details}
                                  </span>
                                ))}
                              </div>
                            )}

                            {/* Action Row */}
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginTop: '0.25rem' }}>
                              <button
                                onClick={() => handleCopy(msg.id, msg.responseData.summary || msg.text)}
                                style={{
                                  background: 'transparent',
                                  border: 'none',
                                  color: copiedId === msg.id ? '#10a37f' : 'var(--text-muted)',
                                  cursor: 'pointer',
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: '0.35rem',
                                  fontSize: '0.75rem',
                                  padding: '4px 6px',
                                  borderRadius: '6px',
                                }}
                                onMouseEnter={(e) => (e.currentTarget.style.color = '#ececec')}
                                onMouseLeave={(e) =>
                                  (e.currentTarget.style.color =
                                    copiedId === msg.id ? '#10a37f' : 'var(--text-muted)')
                                }
                              >
                                {copiedId === msg.id ? <Check size={14} /> : <Copy size={14} />}
                                <span>{copiedId === msg.id ? 'Copied' : 'Copy'}</span>
                              </button>

                              {/* Follow-up shortcut chips */}
                              <button
                                onClick={() =>
                                  handleSendMessage(
                                    'Based on this scene, what heavy equipment and rescue teams should be dispatched first?'
                                  )
                                }
                                style={{
                                  background: 'transparent',
                                  border: '1px solid var(--border-light)',
                                  borderRadius: '9999px',
                                  color: 'var(--text-secondary)',
                                  fontSize: '0.75rem',
                                  padding: '0.2rem 0.65rem',
                                  cursor: 'pointer',
                                }}
                                onMouseEnter={(e) => {
                                  e.currentTarget.style.borderColor = 'var(--border-hover)';
                                  e.currentTarget.style.color = '#fff';
                                }}
                                onMouseLeave={(e) => {
                                  e.currentTarget.style.borderColor = 'var(--border-light)';
                                  e.currentTarget.style.color = 'var(--text-secondary)';
                                }}
                              >
                                ↳ Ask: Required rescue equipment?
                              </button>

                              <button
                                onClick={() =>
                                  handleSendMessage(
                                    'What secondary hazards (fire, utility gas leaks, water contamination) are threatened here?'
                                  )
                                }
                                style={{
                                  background: 'transparent',
                                  border: '1px solid var(--border-light)',
                                  borderRadius: '9999px',
                                  color: 'var(--text-secondary)',
                                  fontSize: '0.75rem',
                                  padding: '0.2rem 0.65rem',
                                  cursor: 'pointer',
                                }}
                                onMouseEnter={(e) => {
                                  e.currentTarget.style.borderColor = 'var(--border-hover)';
                                  e.currentTarget.style.color = '#fff';
                                }}
                                onMouseLeave={(e) => {
                                  e.currentTarget.style.borderColor = 'var(--border-light)';
                                  e.currentTarget.style.color = 'var(--text-secondary)';
                                }}
                              >
                                ↳ Ask: Secondary hazards?
                              </button>
                            </div>
                          </div>
                        ) : (
                          /* Plain Assistant Response */
                          <div>
                            <p style={{ fontSize: '0.95rem', lineHeight: 1.6, color: '#f3f4f6', whiteSpace: 'pre-wrap' }}>
                              {msg.text}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))}

              {/* Waiting / Generating Indicator */}
              {isSending && (
                <div
                  className="animate-fade-in"
                  style={{
                    display: 'flex',
                    gap: '1rem',
                    width: '100%',
                    padding: '0.5rem 0',
                  }}
                >
                  <div
                    style={{
                      width: '32px',
                      height: '32px',
                      borderRadius: '8px',
                      background: 'rgba(255, 255, 255, 0.05)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                      boxShadow: '0 2px 8px rgba(0, 114, 194, 0.2)',
                      overflow: 'hidden',
                    }}
                  >
                    <img
                      src="/logo-icon.png"
                      alt="DisasterMind AI"
                      style={{ width: '22px', height: '22px', objectFit: 'contain' }}
                    />
                  </div>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      flex: 1,
                      gap: '0.75rem',
                      flexWrap: 'wrap',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                      <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                        Reasoning over visual damage & checking live bulletins...
                      </span>
                      <span className="online-dot" />
                    </div>

                    <button
                      type="button"
                      onClick={handleCancelGeneration}
                      title="Stop assessment (Cancel)"
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.45rem',
                        backgroundColor: 'rgba(239, 68, 68, 0.15)',
                        border: '1px solid rgba(239, 68, 68, 0.35)',
                        borderRadius: '6px',
                        color: '#f87171',
                        padding: '0.3rem 0.75rem',
                        fontSize: '0.8rem',
                        fontWeight: 500,
                        cursor: 'pointer',
                        transition: 'all 0.15s ease',
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = 'rgba(239, 68, 68, 0.28)';
                        e.currentTarget.style.borderColor = '#f87171';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = 'rgba(239, 68, 68, 0.15)';
                        e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.35)';
                      }}
                    >
                      <Square size={11} fill="currentColor" />
                      <span>Stop</span>
                    </button>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* ============================================================ */}
        {/* CHAT INPUT BAR (ChatGPT Style)                               */}
        {/* ============================================================ */}
        <div className="chat-input-wrapper">
          <div className="chat-input-box">
            {/* Attached Image Previews in Input Bar */}
            {attachedPreviews.length > 0 && (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.65rem',
                  overflowX: 'auto',
                  paddingBottom: '0.35rem',
                  borderBottom: '1px solid var(--border-light)',
                }}
              >
                {attachedPreviews.map((item, idx) => (
                  <div
                    key={idx}
                    style={{
                      position: 'relative',
                      borderRadius: '8px',
                      overflow: 'hidden',
                      border: '1px solid rgba(255, 255, 255, 0.15)',
                      flexShrink: 0,
                    }}
                  >
                    <img
                      src={item.url}
                      alt={item.name}
                      style={{
                        width: '56px',
                        height: '56px',
                        objectFit: 'cover',
                        display: 'block',
                      }}
                    />
                    <button
                      onClick={() => handleRemoveAttached(idx)}
                      title="Remove image"
                      style={{
                        position: 'absolute',
                        top: '2px',
                        right: '2px',
                        width: '18px',
                        height: '18px',
                        borderRadius: '50%',
                        backgroundColor: 'rgba(0, 0, 0, 0.75)',
                        border: 'none',
                        color: '#fff',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        cursor: 'pointer',
                      }}
                    >
                      <X size={12} />
                    </button>
                  </div>
                ))}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginLeft: 'auto' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    {attachedPreviews.length} image{attachedPreviews.length > 1 ? 's' : ''} ready
                  </span>
                  <button
                    type="button"
                    onClick={clearInput}
                    title="Remove all uploaded images and text"
                    style={{
                      background: 'none',
                      border: 'none',
                      color: 'var(--text-muted)',
                      fontSize: '0.75rem',
                      cursor: 'pointer',
                      padding: '2px 6px',
                      borderRadius: '4px',
                      textDecoration: 'underline',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = '#ef4444')}
                    onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-muted)')}
                  >
                    Clear all
                  </button>
                </div>
              </div>
            )}

            {/* Input Row: Attachment + Textarea + Send */}
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: '0.65rem' }}>
              {/* File Attachment Button */}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                title="Attach disaster images (or drag & drop)"
                style={{
                  width: '36px',
                  height: '36px',
                  borderRadius: '50%',
                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                  border: '1px solid var(--border-light)',
                  color: 'var(--text-secondary)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  flexShrink: 0,
                  transition: 'all 0.15s ease',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.12)';
                  e.currentTarget.style.color = '#ffffff';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
                  e.currentTarget.style.color = 'var(--text-secondary)';
                }}
              >
                <Paperclip size={18} />
              </button>

              {/* Textarea */}
              <textarea
                ref={textareaRef}
                rows={1}
                className="chat-textarea"
                value={inputText}
                onChange={handleTextareaInput}
                onKeyDown={handleKeyDown}
                placeholder={
                  activeSession.messages.length > 0
                    ? 'Ask follow-up question based on this conversation or image...'
                    : 'Upload images or ask whatever you want about disaster assessment...'
                }
              />

              {/* Send or Stop/Cancel Button */}
              {isSending ? (
                <button
                  type="button"
                  onClick={handleCancelGeneration}
                  title="Stop generation (Cancel)"
                  style={{
                    width: '36px',
                    height: '36px',
                    borderRadius: '50%',
                    backgroundColor: '#ef4444',
                    border: 'none',
                    color: '#ffffff',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor: 'pointer',
                    flexShrink: 0,
                    transition: 'all 0.15s ease',
                    boxShadow: '0 0 14px rgba(239, 68, 68, 0.5)',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#dc2626';
                    e.currentTarget.style.transform = 'scale(1.05)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = '#ef4444';
                    e.currentTarget.style.transform = 'scale(1)';
                  }}
                >
                  <Square size={13} fill="#ffffff" />
                </button>
              ) : (
                <button
                  type="button"
                  disabled={!inputText.trim() && attachedFiles.length === 0}
                  onClick={() => handleSendMessage()}
                  title="Send message"
                  style={{
                    width: '36px',
                    height: '36px',
                    borderRadius: '50%',
                    backgroundColor:
                      inputText.trim() || attachedFiles.length > 0
                        ? 'var(--accent-sky)'
                        : 'rgba(255, 255, 255, 0.08)',
                    border: 'none',
                    color:
                      inputText.trim() || attachedFiles.length > 0 ? '#040711' : 'var(--text-muted)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor:
                      inputText.trim() || attachedFiles.length > 0 ? 'pointer' : 'not-allowed',
                    flexShrink: 0,
                    transition: 'all 0.15s ease',
                    boxShadow:
                      inputText.trim() || attachedFiles.length > 0
                        ? '0 0 14px rgba(56, 189, 248, 0.45)'
                        : 'none',
                  }}
                >
                  <Send size={16} />
                </button>
              )}
            </div>
          </div>

          <div
            style={{
              fontSize: '0.72rem',
              color: 'var(--text-muted)',
              textAlign: 'center',
              marginTop: '0.45rem',
            }}
          >
            DisasterMind AI evaluates multimodal imagery and USGS telemetry. Confirm emergency response with certified authorities.
          </div>
        </div>
      </main>

      {/* ============================================================ */}
      {/* IMAGE PREVIEW LIGHTBOX MODAL                                 */}
      {/* ============================================================ */}
      {selectedPreviewImage && (
        <div
          onClick={() => setSelectedPreviewImage(null)}
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.85)',
            backdropFilter: 'blur(8px)',
            zIndex: 100,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '2rem',
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              position: 'relative',
              maxWidth: '90vw',
              maxHeight: '90vh',
              borderRadius: '12px',
              overflow: 'hidden',
              boxShadow: '0 16px 48px rgba(0,0,0,0.8)',
            }}
          >
            <img
              src={selectedPreviewImage}
              alt="Disaster Scene Inspection"
              style={{
                width: '100%',
                height: '100%',
                maxHeight: '85vh',
                objectFit: 'contain',
                display: 'block',
              }}
            />
            <button
              onClick={() => setSelectedPreviewImage(null)}
              style={{
                position: 'absolute',
                top: '12px',
                right: '12px',
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                backgroundColor: 'rgba(0, 0, 0, 0.7)',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                color: '#fff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
              }}
            >
              <X size={18} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
