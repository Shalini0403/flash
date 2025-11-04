import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import PyPDF2
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="Flashcard Generator API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

AVAILABLE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
]

BASELINE_PROMPT = """
You are an educational assistant that reads study material and creates flashcards
to help students remember key facts.

Your task: **generate concise, varied flashcards** in valid JSON format.

Each flashcard must have:
- "front": a short question or prompt (≤120 characters)
- "back": a list of 1–4 short bullets (each ≤1 line)
- "hint": a brief clue if possible (optional but preferred)

Format your final output strictly as:
{{
  "cards": [
    {{
      "front": "...",
      "back": ["...", "..."],
      "hint": "..."
    }},
    ...
  ]
}}

Return only JSON — no extra commentary.

Study Text:
{source_text}
"""


# Pydantic Models
class Flashcard(BaseModel):
    front: str
    back: List[str]
    hint: Optional[str] = None


class PDFSection(BaseModel):
    title: str
    content: str
    page_start: int
    page_end: int
    level: int
    word_count: int
    preview: str


class GenerateRequest(BaseModel):
    study_text: str = Field(..., min_length=50)
    model: str = Field(default="llama-3.1-8b-instant")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    cards: List[Flashcard]
    model_used: str
    card_count: int


class PDFUploadResponse(BaseModel):
    filename: str
    total_pages: int
    total_sections: int
    sections: List[PDFSection]


class HealthResponse(BaseModel):
    status: str
    api_key_configured: bool
    available_models: List[str]


# PDF Processing
CHAPTER_PATTERNS = [
    r'^CHAPTER\s+[IVXLCDM\d]+[\s:.—-]*(.+)$',
    r'^Chapter\s+[IVXLCDM\d]+[\s:.—-]*(.+)$',
    r'^UNIT\s+\d+[\s:.—-]*(.+)$',
    r'^PART\s+[IVXLCDM\d]+[\s:.—-]*(.+)$',
    r'^MODULE\s+\d+[\s:.—-]*(.+)$',
    r'^SECTION\s+[IVXLCDM\d]+[\s:.—-]*(.+)$',
]

def extract_text_from_pdf(pdf_file) -> List[Dict[str, Any]]:
    """Extract text from PDF"""
    pages = []
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(pdf_reader.pages)
        
        for page_num in range(total_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            pages.append({
                'page_number': page_num + 1,
                'text': text
            })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")
    
    return pages


def is_likely_header(line: str, next_lines: List[str]) -> tuple:
    """Check if line is a major header"""
    line = line.strip()
    
    if len(line) < 3 or len(line) > 200:
        return False, 0, None
    
    for pattern in CHAPTER_PATTERNS:
        match = re.match(pattern, line, re.IGNORECASE)
        if match:
            title = match.group(1).strip() if match.groups() else line
            return True, 1, title
    
    words = line.split()
    if line.isupper() and 2 <= len(words) <= 15 and 15 <= len(line) <= 120:
        if next_lines and not next_lines[0].strip().isupper():
            return True, 1, line
    
    if re.match(r'^([IVXLCDM]+|\d{1,2})\s+[A-Z][A-Za-z\s]{10,80}$', line):
        return True, 1, line
    
    return False, 0, None


def clean_text(text: str) -> str:
    """Clean text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
    return text.strip()


def detect_sections(pages: List[Dict[str, Any]]) -> List[PDFSection]:
    """Detect major sections in PDF"""
    sections = []
    current_section = None
    current_content = []
    current_page_start = 1
    
    MIN_SECTION_CHARS = 2000
    MIN_WORD_COUNT = 300
    
    for page_data in pages:
        page_num = page_data['page_number']
        text = page_data['text']
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            next_lines = [lines[j].strip() for j in range(i+1, min(i+5, len(lines)))]
            is_header, level, title = is_likely_header(line, next_lines)
            
            if is_header and title:
                if current_section and current_content:
                    content_text = clean_text(' '.join(current_content))
                    word_count = len(content_text.split())
                    
                    if len(content_text) >= MIN_SECTION_CHARS and word_count >= MIN_WORD_COUNT:
                        preview = content_text[:200] + '...' if len(content_text) > 200 else content_text
                        sections.append(PDFSection(
                            title=current_section,
                            content=content_text,
                            page_start=current_page_start,
                            page_end=page_num - 1 if page_num > 1 else page_num,
                            level=1,
                            word_count=word_count,
                            preview=preview
                        ))
                
                current_section = title
                current_content = []
                current_page_start = page_num
            else:
                current_content.append(line)
    
    # Save last section
    if current_section and current_content:
        content_text = clean_text(' '.join(current_content))
        word_count = len(content_text.split())
        
        if len(content_text) >= MIN_SECTION_CHARS and word_count >= MIN_WORD_COUNT:
            preview = content_text[:200] + '...' if len(content_text) > 200 else content_text
            sections.append(PDFSection(
                title=current_section,
                content=content_text,
                page_start=current_page_start,
                page_end=pages[-1]['page_number'],
                level=1,
                word_count=word_count,
                preview=preview
            ))
    
    # Fallback
    if len(sections) < 2 and pages:
        sections = fallback_section_division(pages, MIN_SECTION_CHARS, MIN_WORD_COUNT)
    
    return sections


def fallback_section_division(pages: List[Dict[str, Any]], min_chars: int, min_words: int) -> List[PDFSection]:
    """Fallback: divide by page clusters"""
    sections = []
    total_pages = len(pages)
    
    if total_pages <= 20:
        cluster_size = 5
    elif total_pages <= 50:
        cluster_size = 8
    elif total_pages <= 100:
        cluster_size = 10
    else:
        cluster_size = 15
    
    section_num = 1
    for i in range(0, len(pages), cluster_size):
        page_group = pages[i:i+cluster_size]
        content = ' '.join([p['text'] for p in page_group])
        content = clean_text(content)
        word_count = len(content.split())
        
        if len(content) >= min_chars and word_count >= min_words:
            sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
            title = None
            
            if sentences:
                for sentence in sentences[:5]:
                    if 10 < len(sentence) < 100 and sentence[0].isupper():
                        title = sentence[:80] + ('...' if len(sentence) > 80 else '')
                        break
            
            if not title:
                title = f"Section {section_num}: Pages {page_group[0]['page_number']}-{page_group[-1]['page_number']}"
            
            preview = content[:200] + '...' if len(content) > 200 else content
            sections.append(PDFSection(
                title=title,
                content=content,
                page_start=page_group[0]['page_number'],
                page_end=page_group[-1]['page_number'],
                level=1,
                word_count=word_count,
                preview=preview
            ))
            section_num += 1
    
    return sections


# Groq Client
def get_groq_client() -> Optional[Groq]:
    if not GROQ_API_KEY:
        return None
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        return None


async def generate_flashcards(
    client: Groq, 
    model: str, 
    study_text: str, 
    temperature: float = 0.7
) -> List[Dict[str, Any]]:
    """Generate flashcards"""
    try:
        prompt = BASELINE_PROMPT.format(source_text=study_text)
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        
        content = response.choices[0].message.content.strip()
        content = re.sub(r"```json\n|```", "", content, flags=re.IGNORECASE).strip()
        
        data = json.loads(content)
        cards = data.get("cards", [])
        
        return cards
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating flashcards: {str(e)}")


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check"""
    return HealthResponse(
        status="healthy",
        api_key_configured=GROQ_API_KEY is not None,
        available_models=AVAILABLE_MODELS
    )


@app.get("/models")
async def get_models():
    """Get available models"""
    return {"models": AVAILABLE_MODELS}


@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF and extract sections"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    try:
        content = await file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            with open(tmp_file_path, 'rb') as pdf_file:
                pages = extract_text_from_pdf(pdf_file)
            
            sections = detect_sections(pages)
            
            return PDFUploadResponse(
                filename=file.filename,
                total_pages=len(pages),
                total_sections=len(sections),
                sections=sections
            )
        
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/generate", response_model=GenerateResponse)
async def generate_cards(request: GenerateRequest):
    """Generate flashcards from text"""
    
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model")
    
    client = get_groq_client()
    if not client:
        raise HTTPException(status_code=500, detail="Failed to initialize client")
    
    cards = await generate_flashcards(client, request.model, request.study_text, request.temperature)
    
    if not cards:
        raise HTTPException(status_code=500, detail="No flashcards generated")
    
    return GenerateResponse(
        cards=cards,
        model_used=request.model,
        card_count=len(cards)
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)