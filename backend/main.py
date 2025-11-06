"""
Flashcard Generator API - Clean Version
A FastAPI application for generating educational flashcards from study materials.
"""

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import PyPDF2
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

# Hugging Face Models Configuration
HUGGINGFACE_MODELS = {
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-1.7b": "Qwen/Qwen3-1.7B",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
}

AVAILABLE_MODELS = list(HUGGINGFACE_MODELS.keys())
DEFAULT_MODEL = "qwen-1.5b"

# Global model cache
hf_model_cache = {}
hf_tokenizer_cache = {}

# PDF section detection parameters
MIN_SECTION_CHARS = 2000
MIN_WORD_COUNT = 300

CHAPTER_PATTERNS = [
    r'^CHAPTER\s+[IVXLCDM\d]+[\s:.—-]*(.+)$',
    r'^Chapter\s+[IVXLCDM\d]+[\s:.—-]*(.+)$',
    r'^UNIT\s+\d+[\s:.—-]*(.+)$',
    r'^PART\s+[IVXLCDM\d]+[\s:.—-]*(.+)$',
    r'^MODULE\s+\d+[\s:.—-]*(.+)$',
    r'^SECTION\s+[IVXLCDM\d]+[\s:.—-]*(.+)$',
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


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

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
    model: str = Field(default=DEFAULT_MODEL)
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
    available_models: List[str]


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

class Logger:
    """Simple logging utility for consistent output"""
    
    @staticmethod
    def header(message: str, width: int = 80):
        """Print a header message"""
        print("\n" + "=" * width)
        print(message)
        print("=" * width)
    
    @staticmethod
    def section(message: str):
        """Print a section message"""
        print(f"\n{message}")
    
    @staticmethod
    def info(message: str, indent: int = 2):
        """Print an info message"""
        print(" " * indent + f"[INFO] {message}")
    
    @staticmethod
    def success(message: str, indent: int = 2):
        """Print a success message"""
        print(" " * indent + f"[OK] {message}")
    
    @staticmethod
    def warning(message: str, indent: int = 2):
        """Print a warning message"""
        print(" " * indent + f"[WARNING] {message}")
    
    @staticmethod
    def error(message: str, indent: int = 2):
        """Print an error message"""
        print(" " * indent + f"[ERROR] {message}")


# ============================================================================
# SYSTEM INITIALIZATION
# ============================================================================

def initialize_system():
    """Initialize and validate system configuration"""
    Logger.header("FLASHCARD GENERATOR API - STARTING UP")
    
    # Check models configuration
    Logger.section("Configuration Check:")
    Logger.success(f"Available models: {AVAILABLE_MODELS}")
    Logger.success(f"Total models configured: {len(AVAILABLE_MODELS)}")
    Logger.success(f"Default model: {DEFAULT_MODEL}")
    
    # Check HF token
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        Logger.success("HF_TOKEN found (for gated models)")
    else:
        Logger.info("HF_TOKEN not set (not needed for public models)")
    
    # Check PyTorch device
    check_pytorch_device()
    
    # Print usage information
    print_usage_info()


def check_pytorch_device():
    """Check and log PyTorch device availability"""
    try:
        if torch.backends.mps.is_available():
            device = "mps (Apple Silicon)"
            Logger.success(f"PyTorch device: {device}")
            Logger.info("Mac M2 detected - MPS acceleration enabled")
        elif torch.cuda.is_available():
            device = "cuda"
            Logger.success(f"PyTorch device: {device}")
            Logger.success(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            Logger.success(f"PyTorch device: {device}")
    except Exception as e:
        Logger.error(f"PyTorch check failed: {e}")


def print_usage_info():
    """Print API usage information"""
    Logger.section("Usage Examples:")
    print('  Python: requests.post("http://localhost:8000/generate", '
          'json={"study_text": "...", "model": "qwen-1.5b"})')
    print('  cURL: curl -X POST "http://localhost:8000/generate" '
          '-H "Content-Type: application/json" '
          '-d \'{"study_text": "...", "model": "qwen-1.5b"}\'')
    
    Logger.section("API Endpoints:")
    print("  - Health check: http://localhost:8000/")
    print("  - Available models: http://localhost:8000/models")
    print("  - Generate endpoint: http://localhost:8000/generate")
    print("  - Upload PDF: http://localhost:8000/upload-pdf")


# Initialize system on module load
initialize_system()


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Flashcard Generator API",
    version="2.0.0",
    description="Generate educational flashcards from study materials"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# PDF PROCESSING MODULE
# ============================================================================

class PDFProcessor:
    """Handles PDF text extraction and section detection"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> List[Dict[str, Any]]:
        """Extract text from PDF file"""
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
            raise HTTPException(
                status_code=400,
                detail=f"Error reading PDF: {str(e)}"
            )
        
        return pages
    
    @staticmethod
    def is_likely_header(line: str, next_lines: List[str]) -> tuple:
        """Check if a line is likely a section header"""
        line = line.strip()
        
        if len(line) < 3 or len(line) > 200:
            return False, 0, None
        
        # Check against known chapter patterns
        for pattern in CHAPTER_PATTERNS:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                title = match.group(1).strip() if match.groups() else line
                return True, 1, title
        
        # Check for all-caps headers
        words = line.split()
        if line.isupper() and 2 <= len(words) <= 15 and 15 <= len(line) <= 120:
            if next_lines and not next_lines[0].strip().isupper():
                return True, 1, line
        
        # Check for numbered sections
        if re.match(r'^([IVXLCDM]+|\d{1,2})\s+[A-Z][A-Za-z\s]{10,80}$', line):
            return True, 1, line
        
        return False, 0, None
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
        return text.strip()
    
    @staticmethod
    def detect_sections(pages: List[Dict[str, Any]]) -> List[PDFSection]:
        """Detect major sections in PDF"""
        sections = []
        current_section = None
        current_content = []
        current_page_start = 1
        
        for page_data in pages:
            page_num = page_data['page_number']
            text = page_data['text']
            lines = text.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                next_lines = [
                    lines[j].strip() 
                    for j in range(i+1, min(i+5, len(lines)))
                ]
                is_header, level, title = PDFProcessor.is_likely_header(
                    line, next_lines
                )
                
                if is_header and title:
                    if current_section and current_content:
                        section = PDFProcessor._create_section(
                            current_section,
                            current_content,
                            current_page_start,
                            page_num - 1 if page_num > 1 else page_num
                        )
                        if section:
                            sections.append(section)
                    
                    current_section = title
                    current_content = []
                    current_page_start = page_num
                else:
                    current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            section = PDFProcessor._create_section(
                current_section,
                current_content,
                current_page_start,
                pages[-1]['page_number']
            )
            if section:
                sections.append(section)
        
        # Fallback if too few sections detected
        if len(sections) < 2 and pages:
            sections = PDFProcessor._fallback_section_division(pages)
        
        return sections
    
    @staticmethod
    def _create_section(
        title: str,
        content: List[str],
        page_start: int,
        page_end: int
    ) -> Optional[PDFSection]:
        """Create a PDFSection object from content"""
        content_text = PDFProcessor.clean_text(' '.join(content))
        word_count = len(content_text.split())
        
        if len(content_text) >= MIN_SECTION_CHARS and word_count >= MIN_WORD_COUNT:
            preview = (
                content_text[:200] + '...' 
                if len(content_text) > 200 
                else content_text
            )
            return PDFSection(
                title=title,
                content=content_text,
                page_start=page_start,
                page_end=page_end,
                level=1,
                word_count=word_count,
                preview=preview
            )
        return None
    
    @staticmethod
    def _fallback_section_division(
        pages: List[Dict[str, Any]]
    ) -> List[PDFSection]:
        """Fallback: divide by page clusters"""
        sections = []
        total_pages = len(pages)
        
        # Determine cluster size based on document length
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
            content = PDFProcessor.clean_text(content)
            word_count = len(content.split())
            
            if len(content) >= MIN_SECTION_CHARS and word_count >= MIN_WORD_COUNT:
                # Try to extract a meaningful title
                title = PDFProcessor._extract_title_from_content(
                    content, section_num, page_group
                )
                
                preview = (
                    content[:200] + '...' 
                    if len(content) > 200 
                    else content
                )
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
    
    @staticmethod
    def _extract_title_from_content(
        content: str,
        section_num: int,
        page_group: List[Dict[str, Any]]
    ) -> str:
        """Extract a title from content or generate a default one"""
        sentences = [
            s.strip() 
            for s in content.split('.') 
            if len(s.strip()) > 20
        ]
        
        if sentences:
            for sentence in sentences[:5]:
                if 10 < len(sentence) < 100 and sentence[0].isupper():
                    return (
                        sentence[:80] + ('...' if len(sentence) > 80 else '')
                    )
        
        return (
            f"Section {section_num}: "
            f"Pages {page_group[0]['page_number']}-{page_group[-1]['page_number']}"
        )


# ============================================================================
# MODEL MANAGEMENT MODULE
# ============================================================================

class ModelManager:
    """Handles loading and caching of Hugging Face models"""
    
    @staticmethod
    def load_model(model_key: str):
        """Load Hugging Face model and tokenizer with caching"""
        global hf_model_cache, hf_tokenizer_cache
        
        # Return cached model if available
        if model_key in hf_model_cache:
            Logger.success(f"Using cached model: {model_key}")
            return hf_model_cache[model_key], hf_tokenizer_cache[model_key]
        
        try:
            model_path = HUGGINGFACE_MODELS[model_key]
            Logger.header(f"Loading model: {model_key}", width=60)
            Logger.info(f"HuggingFace path: {model_path}")
            
            # Get HuggingFace token
            hf_token = os.getenv("HF_TOKEN", None)
            if hf_token:
                Logger.success("Using HF_TOKEN for authentication")
            else:
                Logger.info("No HF_TOKEN (public model)")
            
            # Load tokenizer
            Logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                token=hf_token
            )
            Logger.success("Tokenizer loaded")
            
            # Detect and load model for appropriate device
            device = ModelManager._detect_device()
            model = ModelManager._load_model_for_device(
                model_path, device, hf_token
            )
            
            # Cache the model and tokenizer
            hf_model_cache[model_key] = model
            hf_tokenizer_cache[model_key] = tokenizer
            
            Logger.success(f"Model cached as '{model_key}'")
            print("=" * 60 + "\n")
            
            return model, tokenizer
        
        except Exception as e:
            ModelManager._handle_loading_error(model_key, str(e))
    
    @staticmethod
    def _detect_device() -> str:
        """Detect available compute device"""
        if torch.backends.mps.is_available():
            Logger.info("Using Apple Silicon MPS (Metal Performance Shaders)")
            return "mps"
        elif torch.cuda.is_available():
            Logger.info("Using CUDA GPU")
            return "cuda"
        else:
            Logger.info("Using CPU")
            return "cpu"
    
    @staticmethod
    def _load_model_for_device(model_path: str, device: str, hf_token: str):
        """Load model optimized for specific device"""
        Logger.info("Loading model (may take a few minutes on first run)...")
        
        if device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=hf_token
            )
            model = model.to(device)
            Logger.success("Model loaded with MPS acceleration")
        elif device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=hf_token
            )
            Logger.success("Model loaded with CUDA acceleration")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=hf_token
            )
            model = model.to(device)
            Logger.success("Model loaded on CPU")
        
        return model
    
    @staticmethod
    def _handle_loading_error(model_key: str, error_msg: str):
        """Handle model loading errors"""
        Logger.header(f"ERROR loading model '{model_key}'", width=60)
        Logger.error(error_msg)
        
        if "401" in error_msg or "403" in error_msg:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Authentication error: This model requires a HuggingFace token. "
                    f"Set HF_TOKEN in your .env file or use a different model. "
                    f"Error: {error_msg}"
                )
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading HuggingFace model '{model_key}': {error_msg}"
            )
    
    @staticmethod
    def generate_text(
        model_key: str,
        prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 2048
    ) -> str:
        """Generate text using Hugging Face model"""
        try:
            model, tokenizer = ModelManager.load_model(model_key)
            
            # Format prompt for chat models
            messages = [{"role": "user", "content": prompt}]
            
            # Apply chat template if available
            if hasattr(tokenizer, 'apply_chat_template'):
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                formatted_prompt = prompt
            
            # Tokenize
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt"
            ).to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract generated part (remove prompt)
            if hasattr(tokenizer, 'apply_chat_template'):
                response = generated_text[len(formatted_prompt):].strip()
            else:
                response = generated_text[len(prompt):].strip()
            
            return response
        
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating with HuggingFace: {str(e)}"
            )


# ============================================================================
# JSON PARSING MODULE
# ============================================================================

class JSONParser:
    """Robust JSON parsing with multiple fallback strategies"""
    
    @staticmethod
    def extract_and_fix_json(content: str) -> Dict[str, Any]:
        """Extract and fix JSON from model output"""
        
        # Strategy 1: Remove markdown code blocks
        content = re.sub(
            r"```json\n?|```\n?",
            "",
            content,
            flags=re.IGNORECASE
        ).strip()
        
        # Strategy 2: Try pattern matching
        json_patterns = [
            r'\{[\s\S]*?"cards"[\s\S]*?\}(?=\s*$)',
            r'\{[\s\S]*?"cards"[\s\S]*?\]\s*\}',
            r'\{[\s\S]*?"cards"\s*:\s*\[[\s\S]*?\]\s*\}',
        ]
        
        for pattern in json_patterns:
            json_match = re.search(pattern, content)
            if json_match:
                try:
                    extracted = json_match.group(0)
                    return json.loads(extracted)
                except json.JSONDecodeError:
                    continue
        
        # Strategy 3: Find JSON start
        start_idx = content.find('{')
        if start_idx == -1:
            raise ValueError("No JSON object found in response")
        
        content = content[start_idx:]
        
        # Strategy 4: Extract complete card objects
        cards = JSONParser._extract_complete_cards(content)
        if cards:
            return {"cards": cards}
        
        # Strategy 5: Try parsing as-is
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("Could not extract valid JSON from response")
    
    @staticmethod
    def _extract_complete_cards(content: str) -> List[Dict[str, Any]]:
        """Extract complete card objects from partial JSON"""
        cards_match = re.search(r'"cards"\s*:\s*\[(.*)', content, re.DOTALL)
        if not cards_match:
            return []
        
        cards_content = cards_match.group(1)
        complete_cards = []
        brace_count = 0
        current_card = ""
        in_card = False
        
        for char in cards_content:
            if char == '{':
                brace_count += 1
                in_card = True
            elif char == '}':
                brace_count -= 1
                current_card += char
                if brace_count == 0 and in_card:
                    try:
                        card_obj = json.loads(current_card)
                        if 'front' in card_obj and 'back' in card_obj:
                            complete_cards.append(card_obj)
                    except:
                        pass
                    current_card = ""
                    in_card = False
                continue
            
            if in_card:
                current_card += char
        
        return complete_cards
    
    @staticmethod
    def salvage_partial_cards(content: str) -> List[Dict[str, Any]]:
        """Extract individual cards using regex patterns"""
        cards = []
        
        card_pattern = (
            r'\{\s*"front"\s*:\s*"([^"]+)"\s*,\s*"back"\s*:\s*\[(.*?)\]\s*'
            r'(?:,\s*"hint"\s*:\s*"([^"]*)")?\s*\}'
        )
        
        matches = re.finditer(card_pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                front = match.group(1)
                back_str = match.group(2)
                hint = match.group(3) if match.group(3) else None
                
                back_items = re.findall(r'"([^"]+)"', back_str)
                
                if front and back_items:
                    cards.append({
                        "front": front,
                        "back": back_items,
                        "hint": hint
                    })
            except:
                continue
        
        return cards
    
    @staticmethod
    def validate_cards(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and normalize card data"""
        valid_cards = []
        
        for card in cards:
            if isinstance(card, dict) and 'front' in card and 'back' in card:
                # Ensure back is a list
                if not isinstance(card['back'], list):
                    card['back'] = [str(card['back'])]
                
                # Ensure hint exists
                if 'hint' not in card:
                    card['hint'] = None
                
                valid_cards.append(card)
        
        return valid_cards


# ============================================================================
# FLASHCARD GENERATION MODULE
# ============================================================================

class FlashcardGenerator:
    """Handles flashcard generation with retry logic"""
    
    @staticmethod
    async def generate_flashcards(
        model: str,
        study_text: str,
        temperature: float = 0.7,
        max_retries: int = 2
    ) -> List[Dict[str, Any]]:
        """Generate flashcards with automatic retry and error recovery"""
        
        Logger.header("Generating flashcards", width=60)
        Logger.info(f"Model requested: {model}")
        Logger.info(f"Text length: {len(study_text)} characters")
        Logger.info(f"Temperature: {temperature}")
        
        # Validate model
        if model not in HUGGINGFACE_MODELS:
            Logger.error(f"Invalid model '{model}'")
            Logger.info(f"Available models: {list(HUGGINGFACE_MODELS.keys())}")
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid model '{model}'. "
                    f"Available models: {list(HUGGINGFACE_MODELS.keys())}"
                )
            )
        
        Logger.success(f"Model '{model}' is valid")
        
        last_error = None
        content = ""
        
        # Retry loop
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    Logger.info(f"Retry attempt {attempt}/{max_retries}")
                    retry_temp = min(temperature + 0.1 * attempt, 0.9)
                else:
                    retry_temp = temperature
                
                # Generate content
                prompt = BASELINE_PROMPT.format(source_text=study_text)
                Logger.info("Generating with HuggingFace model...")
                content = ModelManager.generate_text(model, prompt, retry_temp)
                Logger.success(f"Generation complete ({len(content)} chars)")
                
                # Parse and validate
                Logger.info("Parsing response...")
                data = JSONParser.extract_and_fix_json(content)
                cards = data.get("cards", [])
                valid_cards = JSONParser.validate_cards(cards)
                
                if not valid_cards:
                    raise ValueError("No valid flashcards found in response")
                
                Logger.success(f"Successfully generated {len(valid_cards)} flashcards")
                print("=" * 60 + "\n")
                
                return valid_cards
            
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                Logger.warning(f"Parse error (attempt {attempt + 1}): {str(e)}")
                Logger.info(f"Content preview: {content[:300]}...")
                
                if attempt < max_retries:
                    continue
                else:
                    # Final attempt: try to salvage
                    Logger.info("Attempting to salvage partial cards...")
                    salvaged_cards = JSONParser.salvage_partial_cards(content)
                    if salvaged_cards:
                        Logger.success(f"Salvaged {len(salvaged_cards)} cards")
                        return salvaged_cards
                    raise
            
            except HTTPException:
                raise
            except Exception as e:
                last_error = e
                Logger.error(f"Generation error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries:
                    continue
                raise
        
        # All retries failed
        Logger.error("All retry attempts failed")
        print("=" * 60 + "\n")
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to generate valid flashcards after {max_retries + 1} attempts. "
                f"Last error: {str(last_error)}"
            )
        )


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        available_models=AVAILABLE_MODELS
    )


@app.get("/models")
async def get_models():
    """Get available models"""
    return {
        "models": AVAILABLE_MODELS,
        "huggingface_models": list(HUGGINGFACE_MODELS.keys())
    }


@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF and extract sections"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files allowed"
        )
    
    try:
        content = await file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            with open(tmp_file_path, 'rb') as pdf_file:
                pages = PDFProcessor.extract_text_from_pdf(pdf_file)
            
            sections = PDFProcessor.detect_sections(pages)
            
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
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )


@app.post("/generate", response_model=GenerateResponse)
async def generate_cards(request: GenerateRequest):
    """Generate flashcards from text"""
    
    Logger.header("NEW REQUEST: /generate")
    Logger.info(f"Requested model: '{request.model}'")
    Logger.info(f"Temperature: {request.temperature}")
    Logger.info(f"Text length: {len(request.study_text)} chars")
    
    if request.model not in AVAILABLE_MODELS:
        Logger.error("Invalid model")
        Logger.info(f"Requested: '{request.model}'")
        Logger.info(f"Available: {AVAILABLE_MODELS}")
        print("=" * 80 + "\n")
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid model '{request.model}'. "
                f"Available models: {AVAILABLE_MODELS}"
            )
        )
    
    Logger.success("Model is valid")
    
    try:
        cards = await FlashcardGenerator.generate_flashcards(
            request.model,
            request.study_text,
            request.temperature
        )
        
        if not cards:
            Logger.warning("No flashcards generated")
            raise HTTPException(
                status_code=500,
                detail="No flashcards generated"
            )
        
        Logger.success(f"Generated {len(cards)} flashcards")
        print("=" * 80 + "\n")
        
        return GenerateResponse(
            cards=cards,
            model_used=request.model,
            card_count=len(cards)
        )
    except HTTPException:
        raise
    except Exception as e:
        Logger.error(f"Error in generate endpoint: {str(e)}")
        print("=" * 80 + "\n")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STARTUP TESTS
# ============================================================================

def run_startup_tests():
    """Run startup validation tests"""
    Logger.header("RUNNING STARTUP TESTS")
    
    # Test 1: Check models configuration
    Logger.section("1. Checking model configuration...")
    if len(AVAILABLE_MODELS) == 0:
        Logger.error("No models configured!")
        Logger.info("Please add models to HUGGINGFACE_MODELS dictionary")
        sys.exit(1)
    else:
        Logger.success(f"{len(AVAILABLE_MODELS)} models configured:")
        for model_key, model_path in HUGGINGFACE_MODELS.items():
            print(f"      - {model_key} -> {model_path}")
    
    # Test 2: Check dependencies
    Logger.section("2. Checking dependencies...")
    try:
        import torch
        import transformers
        Logger.success(f"PyTorch {torch.__version__}")
        Logger.success(f"Transformers {transformers.__version__}")
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        Logger.success(f"Device: {device}")
    except ImportError as e:
        Logger.error(f"Missing dependency: {e}")
        Logger.info("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Test 3: Configuration summary
    Logger.section("3. Configuration summary...")
    Logger.success(f"Default model: {DEFAULT_MODEL}")
    Logger.success("API endpoints: /, /models, /generate, /upload-pdf")
    Logger.success("CORS enabled for: localhost:3000, localhost:5173")
    Logger.success("JSON parsing: Enhanced with 5 fallback strategies")
    Logger.success("Retry logic: Up to 3 attempts with salvage mode")
    
    Logger.header("ALL STARTUP TESTS PASSED")
    
    Logger.section("Quick Test Commands:")
    print("   curl http://localhost:8000/models")
    print('   curl -X POST http://localhost:8000/generate \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"study_text":"Test","model":"qwen-1.5b"}\'')
    
    Logger.header("STARTING SERVER...")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_startup_tests()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )