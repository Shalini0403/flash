import axios from 'axios';
import { motion } from 'framer-motion';
import React, { useEffect, useState } from 'react';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [studyText, setStudyText] = useState('');
  const [selectedModel, setSelectedModel] = useState('llama-3.1-8b-instant');
  const [availableModels, setAvailableModels] = useState([]);
  const [cards, setCards] = useState([]);
  const [currentCardIndex, setCurrentCardIndex] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);
  const [showHint, setShowHint] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState(null);
  
  // PDF related states
  const [pdfFile, setPdfFile] = useState(null);
  const [pdfSections, setPdfSections] = useState([]);
  const [selectedSection, setSelectedSection] = useState(null);
  const [uploadingPdf, setUploadingPdf] = useState(false);
  const [viewMode, setViewMode] = useState('text'); // 'text' or 'pdf'

  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/`);
      setApiStatus(response.data);
      setAvailableModels(response.data.available_models);
    } catch (err) {
      setError('Cannot connect to backend. Make sure it\'s running on port 8000.');
    }
  };

  const handlePdfUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.endsWith('.pdf')) {
      setError('Please upload a PDF file');
      return;
    }

    setPdfFile(file);
    setUploadingPdf(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/upload-pdf`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setPdfSections(response.data.sections);
      setViewMode('pdf');
      setUploadingPdf(false);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to process PDF');
      setUploadingPdf(false);
    }
  };

  const selectSection = (section) => {
    setSelectedSection(section);
    setStudyText(section.content);
  };

  const generateFlashcards = async () => {
    if (studyText.length < 50) {
      setError('Please provide at least 50 characters of study material.');
      return;
    }

    setLoading(true);
    setError(null);
    setCards([]);

    try {
      const response = await axios.post(`${API_BASE_URL}/generate`, {
        study_text: studyText,
        model: selectedModel,
        temperature: 0.7
      });

      setCards(response.data.cards);
      setCurrentCardIndex(0);
      setIsFlipped(false);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate flashcards');
    } finally {
      setLoading(false);
    }
  };

  const flipCard = () => {
    setIsFlipped(!isFlipped);
  };

  const nextCard = () => {
    if (currentCardIndex < cards.length - 1) {
      setCurrentCardIndex(currentCardIndex + 1);
      setIsFlipped(false);
      setShowHint(false);
    }
  };

  const previousCard = () => {
    if (currentCardIndex > 0) {
      setCurrentCardIndex(currentCardIndex - 1);
      setIsFlipped(false);
      setShowHint(false);
    }
  };

  const downloadCards = () => {
    const dataStr = JSON.stringify({ cards }, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'flashcards.json';
    link.click();
  };

  const resetAll = () => {
    setPdfFile(null);
    setPdfSections([]);
    setSelectedSection(null);
    setStudyText('');
    setCards([]);
    setViewMode('text');
    setError(null);
  };

  const currentCard = cards[currentCardIndex];

  return (
    <div className="app">
      <header className="header">
        <h1>🧠 AI Flashcard Generator</h1>
        <p>Powered by Groq LLM</p>
        {apiStatus && (
          <div className={`status ${apiStatus.api_key_configured ? 'success' : 'warning'}`}>
            {apiStatus.api_key_configured ? '✓ Connected' : '⚠ API Key Missing'}
          </div>
        )}
      </header>

      <div className="container">
        <aside className="sidebar">
          {/* View Mode Selector */}
          <div className="view-mode-selector">
            <button
              className={`mode-btn ${viewMode === 'text' ? 'active' : ''}`}
              onClick={() => setViewMode('text')}
            >
              📝 Text Input
            </button>
            <button
              className={`mode-btn ${viewMode === 'pdf' ? 'active' : ''}`}
              onClick={() => setViewMode('pdf')}
            >
              📄 PDF Upload
            </button>
          </div>

          {viewMode === 'text' ? (
            <>
              <div className="input-section">
                <h2>Study Material</h2>
                <textarea
                  value={studyText}
                  onChange={(e) => setStudyText(e.target.value)}
                  placeholder="Paste your study text here..."
                  rows={12}
                />
                <div className="char-count">
                  {studyText.length} characters {studyText.length >= 50 ? '✓' : '(min 50)'}
                </div>
              </div>
            </>
          ) : (
            <>
              <div className="pdf-upload-section">
                <h2>Upload PDF</h2>
                <div className="upload-area">
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={handlePdfUpload}
                    id="pdf-upload"
                    style={{ display: 'none' }}
                  />
                  <label htmlFor="pdf-upload" className="upload-label">
                    {pdfFile ? (
                      <>
                        <span className="file-icon">📄</span>
                        <span className="file-name">{pdfFile.name}</span>
                        <span className="change-file">Click to change</span>
                      </>
                    ) : (
                      <>
                        <span className="upload-icon">📁</span>
                        <span>Click to upload PDF</span>
                        <span className="upload-hint">or drag and drop</span>
                      </>
                    )}
                  </label>
                </div>

                {uploadingPdf && (
                  <div className="processing-pdf">
                    <div className="spinner-small"></div>
                    <span>Processing PDF...</span>
                  </div>
                )}

                {pdfSections.length > 0 && (
                  <div className="sections-list">
                    <h3>Sections Found ({pdfSections.length})</h3>
                    <div className="sections-scroll">
                      {pdfSections.map((section, index) => (
                        <div
                          key={index}
                          className={`section-item ${selectedSection === section ? 'selected' : ''}`}
                          onClick={() => selectSection(section)}
                        >
                          <div className="section-title">{section.title}</div>
                          <div className="section-meta">
                            Pages {section.page_start}-{section.page_end} · {section.word_count} words
                          </div>
                          <div className="section-preview">{section.preview}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {pdfSections.length > 0 && (
                  <button className="reset-btn" onClick={resetAll}>
                    🔄 Reset & Upload New PDF
                  </button>
                )}
              </div>
            </>
          )}

          <div className="model-section">
            <h2>Model Selection</h2>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={loading}
            >
              {availableModels.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </div>

          <button
            className="generate-btn"
            onClick={generateFlashcards}
            disabled={loading || studyText.length < 50}
          >
            {loading ? '⏳ Generating...' : '🚀 Generate Flashcards'}
          </button>

          {cards.length > 0 && (
            <button className="download-btn" onClick={downloadCards}>
              ⬇️ Download JSON
            </button>
          )}

          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}
        </aside>

        <main className="main-content">
          {loading && (
            <div className="loading-state">
              <div className="spinner"></div>
              <p>Generating your flashcards...</p>
            </div>
          )}

          {!loading && cards.length === 0 && !error && (
            <div className="empty-state">
              <div className="empty-icon">📚</div>
              <h2>Welcome!</h2>
              {viewMode === 'pdf' ? (
                <p>Upload a PDF, select a section, and click "Generate" to create flashcards.</p>
              ) : (
                <p>Paste your study material and click "Generate" to create flashcards.</p>
              )}
            </div>
          )}

          {!loading && cards.length > 0 && currentCard && (
            <div className="flashcard-section">
              <div className="card-header">
                <h2>Flashcard {currentCardIndex + 1} of {cards.length}</h2>
                {selectedSection && (
                  <p className="section-label">From: {selectedSection.title}</p>
                )}
              </div>

              <div className="flashcard-container" onClick={flipCard}>
                <motion.div
                  className="flashcard"
                  initial={false}
                  animate={{ rotateY: isFlipped ? 180 : 0 }}
                  transition={{ duration: 0.6, type: 'spring' }}
                >
                  <div className="card-face card-front">
                    <div className="card-content">
                      <p className="question">{currentCard.front}</p>
                      <small className="tap-hint">Click card to reveal answer</small>
                      {currentCard.hint && (
                        <div className="hint-section">
                          {!showHint ? (
                            <button 
                              className="hint-btn"
                              onClick={(e) => {
                                e.stopPropagation();
                                setShowHint(true);
                              }}
                            >
                              💡 Show Hint
                            </button>
                          ) : (
                            <div className="hint-revealed">
                              <strong>💡 Hint:</strong> {currentCard.hint}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="card-face card-back">
                    <div className="card-content">
                      <h3>Answer:</h3>
                      <ul>
                        {currentCard.back.map((bullet, idx) => (
                          <li key={idx}>{bullet}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </motion.div>
              </div>

              <div className="card-navigation">
                <button
                  onClick={previousCard}
                  disabled={currentCardIndex === 0}
                  className="nav-btn"
                >
                  ⬅️ Previous
                </button>

                <div className="card-indicator">
                  {cards.map((_, idx) => (
                    <span
                      key={idx}
                      className={`dot ${idx === currentCardIndex ? 'active' : ''}`}
                      onClick={() => {
                        setCurrentCardIndex(idx);
                        setIsFlipped(false);
                        setShowHint(false);
                      }}
                    />
                  ))}
                </div>

                <button
                  onClick={nextCard}
                  disabled={currentCardIndex === cards.length - 1}
                  className="nav-btn"
                >
                  Next ➡️
                </button>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;