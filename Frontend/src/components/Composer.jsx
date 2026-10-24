import React, { useRef, useEffect } from 'react';
import {
  Send,
  Paperclip,
  Image as ImageIcon,
  X,
  Square,
  RefreshCw,
  Globe,
} from 'lucide-react';

export default function Composer({
  inputText,
  setInputText,
  onSendMessage,
  attachedFiles,
  attachedPreviews,
  onAddFiles,
  onRemoveAttached,
  isSending,
  onCancelGeneration,
  searchLiveWeb,
}) {
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-resize textarea height as user types
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollH = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = Math.min(scrollH, 160) + 'px';
    }
  }, [inputText]);

  const handleKeyDown = (e) => {
    if (e.key === 'Escape' && isSending) {
      e.preventDefault();
      onCancelGeneration();
      return;
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isSending && (inputText.trim() || attachedFiles.length > 0)) {
        onSendMessage();
      }
    }
  };

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      onAddFiles(files);
    }
    e.target.value = '';
  };

  const canSubmit = Boolean(inputText.trim() || attachedFiles.length > 0);

  return (
    <div className="composer-wrapper">
      <div className="composer-box">
        {/* Attached Previews Strip */}
        {attachedPreviews.length > 0 && (
          <div className="composer-preview-strip">
            {attachedPreviews.map((item, idx) => (
              <div key={idx} className="preview-thumbnail">
                <img src={item.url} alt={item.name} />
                <button
                  type="button"
                  onClick={() => onRemoveAttached(idx)}
                  title="Remove image"
                  className="preview-remove-btn"
                >
                  <X size={11} />
                </button>
              </div>
            ))}
            <span style={{ fontSize: '0.74rem', color: 'var(--color-secondary-text)', marginLeft: 'auto' }}>
              {attachedPreviews.length} scene image{attachedPreviews.length > 1 ? 's' : ''} attached
            </span>
          </div>
        )}

        {/* Text Input Row */}
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: '0.65rem' }}>
          {/* File Attachment Action */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/jpeg,image/png,image/webp,image/tiff"
            multiple
            style={{ display: 'none' }}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            title="Attach disaster or scene image"
            className="composer-action-btn"
          >
            <Paperclip size={18} />
          </button>

          {/* Text Area */}
          <textarea
            ref={textareaRef}
            rows={1}
            placeholder="Ask about a disaster scene, location, route from your city, or travel safety..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            className="composer-textarea"
          />

          {/* Submit / Stop Action */}
          {isSending ? (
            <button
              type="button"
              onClick={onCancelGeneration}
              title="Stop investigation"
              className="composer-stop-btn"
            >
              <Square size={13} fill="currentColor" />
            </button>
          ) : (
            <button
              type="button"
              disabled={!canSubmit}
              onClick={() => {
                if (canSubmit) onSendMessage();
              }}
              title="Send question (Enter)"
              className="composer-send-btn"
              style={{
                opacity: canSubmit ? 1 : 0.45,
                cursor: canSubmit ? 'pointer' : 'default',
              }}
            >
              <Send size={15} />
            </button>
          )}
        </div>

        {/* Bottom Helper Bar */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.72rem', color: 'var(--color-secondary-text)', paddingTop: '0.15rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: searchLiveWeb ? 'var(--color-success)' : 'var(--color-secondary-text)' }} />
            <span>{searchLiveWeb ? 'Live USGS & GDACS feeds active' : 'Internal corpus only'}</span>
          </div>
          <span className="hide-on-mobile">
            Press <strong style={{ color: 'var(--color-chocolate)' }}>Enter</strong> to send, <strong style={{ color: 'var(--color-chocolate)' }}>Shift+Enter</strong> for newline
          </span>
        </div>
      </div>
    </div>
  );
}
