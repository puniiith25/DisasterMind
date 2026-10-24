import React, { useState, useEffect, useRef } from 'react';
import {
  PanelLeft,
  Plus,
  RefreshCw,
  X,
  Square,
  AlertTriangle,
} from 'lucide-react';

import DisasterMindEmblem from './components/DisasterMindEmblem';
import Sidebar from './components/Sidebar';
import WelcomeScreen from './components/WelcomeScreen';
import AssistantMessage from './components/AssistantMessage';
import Composer from './components/Composer';
import { executeConversationalInquiry } from './utils/conversationIntelligence';

const STORAGE_KEY = 'disastermind_chat_sessions_v2';

export default function App() {
  // Responsive mobile state
  const [isMobile, setIsMobile] = useState(() => {
    return typeof window !== 'undefined' ? window.innerWidth < 768 : false;
  });

  // Sidebar state
  const [sidebarOpen, setSidebarOpen] = useState(() => {
    return typeof window !== 'undefined' ? window.innerWidth >= 768 : true;
  });

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (mobile) setSidebarOpen(false);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // System status
  const [systemStatus, setSystemStatus] = useState({
    points: 100,
    device: 'mps',
    vlmReady: true,
  });
  const [searchLiveWeb, setSearchLiveWeb] = useState(true);
  const [selectedPreviewImage, setSelectedPreviewImage] = useState(null);

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
        title: 'New Investigation',
        createdAt: new Date().toISOString(),
        image_paths: [],
        confirmed_location: null,
        confirmed_coords: null,
        route_context: null,
        messages: [],
      },
    ];
  });

  const [activeSessionId, setActiveSessionId] = useState(() => {
    return sessions[0]?.id || 'session-1';
  });

  // Save sessions to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    } catch (e) {
      console.warn('Failed to save sessions to localStorage:', e);
    }
  }, [sessions]);

  // Lookup active session
  const activeSession = sessions.find((s) => s.id === activeSessionId) || sessions[0] || {
    id: 'session-default',
    title: 'New Investigation',
    image_paths: [],
    messages: [],
  };

  // Composer Input & Attachments
  const [inputText, setInputText] = useState('');
  const [attachedFiles, setAttachedFiles] = useState([]);
  const [attachedPreviews, setAttachedPreviews] = useState([]);
  const [isSending, setIsSending] = useState(false);

  // Refs
  const messagesEndRef = useRef(null);
  const chatScrollContainerRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeSession?.messages?.length, isSending]);

  // Global Keyboard Shortcuts (Cmd+K for new investigation)
  useEffect(() => {
    const handleGlobalKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        handleNewChat();
      }
    };
    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, []);

  // Fetch initial system telemetry
  useEffect(() => {
    fetch('/health')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) {
          setSystemStatus((prev) => ({
            ...prev,
            points: data.qdrant_points || 100,
          }));
        }
      })
      .catch(() => {});
  }, []);

  // Session Handlers
  const handleNewChat = () => {
    const newSessionId = 'session-' + Date.now();
    const newSession = {
      id: newSessionId,
      title: 'New Investigation',
      createdAt: new Date().toISOString(),
      image_paths: [],
      confirmed_location: null,
      confirmed_coords: null,
      route_context: null,
      messages: [],
    };
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newSessionId);
    setInputText('');
    setAttachedFiles([]);
    setAttachedPreviews([]);
    if (isMobile) setSidebarOpen(false);
  };

  const handleSelectSession = (id) => {
    setActiveSessionId(id);
    setInputText('');
    setAttachedFiles([]);
    setAttachedPreviews([]);
    if (isMobile) setSidebarOpen(false);
  };

  const handleDeleteSession = (id) => {
    setSessions((prev) => {
      const filtered = prev.filter((s) => s.id !== id);
      if (filtered.length === 0) {
        const fresh = {
          id: 'session-' + Date.now(),
          title: 'New Investigation',
          createdAt: new Date().toISOString(),
          image_paths: [],
          confirmed_location: null,
          confirmed_coords: null,
          route_context: null,
          messages: [],
        };
        setActiveSessionId(fresh.id);
        return [fresh];
      }
      if (activeSessionId === id) {
        setActiveSessionId(filtered[0].id);
      }
      return filtered;
    });
  };

  const handleClearAllSessions = () => {
    const fresh = {
      id: 'session-' + Date.now(),
      title: 'New Investigation',
      createdAt: new Date().toISOString(),
      image_paths: [],
      confirmed_location: null,
      confirmed_coords: null,
      route_context: null,
      messages: [],
    };
    setSessions([fresh]);
    setActiveSessionId(fresh.id);
  };

  // Attachment Handlers
  const handleAddFiles = (newFiles) => {
    const validFiles = newFiles.filter(
      (f) =>
        f.type.startsWith('image/') ||
        f.name.match(/\.(jpg|jpeg|png|webp|tiff)$/i)
    );
    if (validFiles.length === 0) return;

    const newPreviews = validFiles.map((file) => ({
      name: file.name,
      url: URL.createObjectURL(file),
    }));

    setAttachedFiles((prev) => [...prev, ...validFiles]);
    setAttachedPreviews((prev) => [...prev, ...newPreviews]);
  };

  const handleRemoveAttached = (index) => {
    setAttachedFiles((prev) => prev.filter((_, i) => i !== index));
    setAttachedPreviews((prev) => {
      const removed = prev[index];
      if (removed?.url) URL.revokeObjectURL(removed.url);
      return prev.filter((_, i) => i !== index);
    });
  };

  const handleCancelGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  // Unified Conversational Message Handler
  const handleSendMessage = async (customPrompt) => {
    const questionText = (customPrompt || inputText).trim();
    if (!questionText && attachedFiles.length === 0) return;

    const userMessageId = 'msg-' + Date.now();
    const assistantMessageId = 'msg-' + (Date.now() + 1);

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

    // Optimistically update session messages and title
    setSessions((prevSessions) =>
      prevSessions.map((session) => {
        if (session.id === activeSession.id) {
          const updatedMessages = [...session.messages, newUserMessage];
          const newTitle =
            session.messages.length === 0
              ? questionText.slice(0, 36) || 'Disaster Assessment'
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

    const filesToSend = [...attachedFiles];
    setInputText('');
    setAttachedFiles([]);
    setAttachedPreviews([]);

    const controller = new AbortController();
    abortControllerRef.current = controller;
    setIsSending(true);

    try {
      // Build conversation history text
      const historyText = activeSession.messages
        .slice(-6)
        .map((m) => {
          if (m.role === 'user') return `User: ${m.text}`;
          const b = m.responseData?.breakdown;
          const ans = b
            ? `${b.what_happened} ${b.impact_and_severity}`
            : m.responseData?.summary || m.text;
          return `DisasterMind: ${ans}`;
        })
        .join('\n');

      const context = {
        image_paths: activeSession.image_paths || [],
        confirmed_location: activeSession.confirmed_location || null,
        confirmed_coords: activeSession.confirmed_coords || null,
        route_context: activeSession.route_context || null,
        historyText,
      };

      // Execute unified intelligence inquiry
      const results = await executeConversationalInquiry({
        question: questionText || 'Analyze the uploaded disaster imagery.',
        files: filesToSend,
        context,
        searchLive: searchLiveWeb,
        engineMode: 'deep_vlm',
        signal: controller.signal,
      });

      const newAssistantMessage = {
        id: assistantMessageId,
        role: 'assistant',
        text: results.vlmData?.summary || results.routeData?.safety_advisory || results.locationData?.direct_answer || 'Analysis complete.',
        responseData: results,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setSessions((prevSessions) =>
        prevSessions.map((session) => {
          if (session.id === activeSession.id) {
            const combinedPaths = Array.from(
              new Set([
                ...(session.image_paths || []),
                ...(results.vlmData?.image_paths || []),
              ])
            );
            return {
              ...session,
              image_paths: combinedPaths,
              confirmed_location: results.establishedLocation || session.confirmed_location,
              confirmed_coords: results.establishedCoords || session.confirmed_coords,
              route_context: results.updatedRouteContext || session.route_context,
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
          text: 'Investigation paused by user.',
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
        text: `Analysis failed: ${err.message}. Please verify the DisasterMind backend service is active.`,
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

  return (
    <div className="workspace-container">
      {/* Lightbox Modal */}
      {selectedPreviewImage && (
        <div className="lightbox-overlay" onClick={() => setSelectedPreviewImage(null)}>
          <div style={{ position: 'relative', maxWidth: '90vw', maxHeight: '90vh' }}>
            <img
              src={selectedPreviewImage}
              alt="Enlarged Preview"
              style={{
                maxWidth: '90vw',
                maxHeight: '90vh',
                borderRadius: '12px',
                border: '1px solid rgba(255,255,255,0.25)',
                boxShadow: '0 20px 60px rgba(40,28,24,0.7)',
              }}
            />
            <button
              onClick={() => setSelectedPreviewImage(null)}
              style={{
                position: 'absolute',
                top: '-12px',
                right: '-12px',
                backgroundColor: '#38251F',
                border: '1px solid #E7D9C9',
                color: '#FFFFFF',
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
              }}
            >
              <X size={16} />
            </button>
          </div>
        </div>
      )}

      {/* Mobile Drawer Backdrop */}
      {isMobile && sidebarOpen && (
        <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Sidebar Navigation */}
      <Sidebar
        sidebarOpen={sidebarOpen}
        onCloseSidebar={() => setSidebarOpen(false)}
        sessions={sessions}
        activeSessionId={activeSession.id}
        onSelectSession={handleSelectSession}
        onNewSession={handleNewChat}
        onDeleteSession={handleDeleteSession}
        onClearAllSessions={handleClearAllSessions}
        systemStatus={systemStatus}
        searchLiveWeb={searchLiveWeb}
        onToggleSearchLiveWeb={() => setSearchLiveWeb((prev) => !prev)}
      />

      {/* Main Workspace Canvas */}
      <main className="workspace-main">
        {/* Top Header */}
        <header className="workspace-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            {!sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(true)}
                title="Open sidebar"
                className="icon-btn"
              >
                <PanelLeft size={18} />
              </button>
            )}

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem' }}>
              <span style={{ fontSize: '0.92rem', fontWeight: 700, color: 'var(--color-chocolate)', fontFamily: 'var(--font-heading)' }}>
                {activeSession.title || 'Disaster Intelligence Workspace'}
              </span>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            {activeSession?.confirmed_location && (
              <span className="badge badge-orange hide-on-mobile">
                📍 {activeSession.confirmed_location}
              </span>
            )}

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.76rem', color: 'var(--color-secondary-text)' }}>
              <span style={{ width: '7px', height: '7px', borderRadius: '50%', backgroundColor: 'var(--color-success)' }} />
              <span className="hide-on-mobile">Corpus Active</span>
            </div>

            <button
              onClick={handleNewChat}
              style={{
                backgroundColor: '#FFFFFF',
                border: '1px solid var(--color-subtle-border)',
                borderRadius: '8px',
                color: 'var(--color-chocolate)',
                padding: '0.35rem 0.75rem',
                fontSize: '0.8rem',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
                boxShadow: 'var(--shadow-subtle)',
                transition: 'all 0.15s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'var(--color-burnt-orange)';
                e.currentTarget.style.color = 'var(--color-burnt-orange)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = 'var(--color-subtle-border)';
                e.currentTarget.style.color = 'var(--color-chocolate)';
              }}
            >
              <Plus size={14} />
              <span>New</span>
            </button>
          </div>
        </header>

        {/* Messages Scroll Area */}
        <div className="messages-container" ref={chatScrollContainerRef}>
          {activeSession.messages.length === 0 ? (
            <WelcomeScreen onSelectPrompt={(prompt) => handleSendMessage(prompt)} />
          ) : (
            <div className="messages-inner">
              {activeSession.messages.map((msg) => (
                <div key={msg.id} className="animate-fade-in" style={{ width: '100%' }}>
                  {msg.role === 'user' ? (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', width: '100%' }}>
                      <div className="user-msg-bubble">
                        {/* Attached Image Thumbnails */}
                        {msg.images && msg.images.length > 0 && (
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem', marginBottom: '0.65rem' }}>
                            {msg.images.map((img, idx) => (
                              <div
                                key={idx}
                                onClick={() => setSelectedPreviewImage(img.url)}
                                style={{
                                  position: 'relative',
                                  borderRadius: '8px',
                                  overflow: 'hidden',
                                  border: '1px solid var(--color-subtle-border)',
                                  cursor: 'pointer',
                                  boxShadow: 'var(--shadow-subtle)',
                                }}
                              >
                                <img
                                  src={img.url}
                                  alt={img.name}
                                  style={{
                                    width: '120px',
                                    height: '80px',
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
                                    fontSize: '0.66rem',
                                    backgroundColor: 'rgba(56, 37, 31, 0.85)',
                                    color: '#ffffff',
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

                        <p style={{ fontSize: '0.96rem', lineHeight: 1.6, whiteSpace: 'pre-wrap', margin: 0 }}>
                          {msg.text}
                        </p>
                        <div style={{ fontSize: '0.7rem', color: 'var(--color-secondary-text)', textAlign: 'right', marginTop: '0.35rem' }}>
                          {msg.timestamp}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <AssistantMessage
                      msg={msg}
                      onFollowUp={(query) => handleSendMessage(query)}
                      onCalculateRoute={(origin, destination) =>
                        handleSendMessage(`What is the route and hazard assessment from ${origin} to ${destination}?`)
                      }
                    />
                  )}
                </div>
              ))}

              {/* Generating Indicator */}
              {isSending && (
                <div style={{ display: 'flex', gap: '1rem', width: '100%', padding: '0.5rem 0', alignItems: 'center' }}>
                  <div style={{ flexShrink: 0 }}>
                    <DisasterMindEmblem size={32} />
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                    <RefreshCw size={15} color="var(--color-burnt-orange)" className="animate-spin" />
                    <span style={{ fontSize: '0.88rem', color: 'var(--color-secondary-text)' }}>
                      Correlating scene evidence, live disaster bulletins & route corridors...
                    </span>
                    <button
                      type="button"
                      onClick={handleCancelGeneration}
                      style={{
                        backgroundColor: 'var(--color-critical-bg)',
                        border: '1px solid var(--color-critical-border)',
                        borderRadius: '6px',
                        color: 'var(--color-critical)',
                        padding: '0.22rem 0.6rem',
                        fontSize: '0.76rem',
                        fontWeight: 600,
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.3rem',
                      }}
                    >
                      <Square size={10} fill="currentColor" />
                      <span>Stop</span>
                    </button>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Floating Bottom Composer */}
        <Composer
          inputText={inputText}
          setInputText={setInputText}
          onSendMessage={() => handleSendMessage()}
          attachedFiles={attachedFiles}
          attachedPreviews={attachedPreviews}
          onAddFiles={handleAddFiles}
          onRemoveAttached={handleRemoveAttached}
          isSending={isSending}
          onCancelGeneration={handleCancelGeneration}
          searchLiveWeb={searchLiveWeb}
        />
      </main>
    </div>
  );
}
