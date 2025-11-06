"""
PDF Processing Module
Handles PDF text extraction, section detection, and content chunking
"""

import re
from typing import Any, Dict, List, Optional

import PyPDF2
from fastapi import HTTPException
from pydantic import BaseModel

# ============================================================================
# CONFIGURATION
# ============================================================================

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


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PDFSection(BaseModel):
    """Represents a detected section in a PDF document"""
    title: str
    content: str
    page_start: int
    page_end: int
    level: int
    word_count: int
    preview: str


# ============================================================================
# PDF PROCESSOR CLASS
# ============================================================================

class PDFProcessor:
    """Handles PDF text extraction and section detection"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> List[Dict[str, Any]]:
        """
        Extract text from PDF file page by page.
        
        Args:
            pdf_file: File-like object containing PDF data
            
        Returns:
            List of dictionaries containing page number and text for each page
            
        Raises:
            HTTPException: If PDF cannot be read or processed
        """
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
        """
        Check if a line is likely a section header.
        
        Args:
            line: The line to check
            next_lines: Following lines for context
            
        Returns:
            Tuple of (is_header: bool, level: int, title: Optional[str])
        """
        line = line.strip()
        
        # Length constraints for headers
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
        """
        Clean and normalize text by removing extra whitespace and page numbers.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        # Remove "Page X" markers
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
        return text.strip()
    
    @staticmethod
    def detect_sections(pages: List[Dict[str, Any]]) -> List[PDFSection]:
        """
        Detect major sections in PDF by analyzing headers and structure.
        
        Args:
            pages: List of page dictionaries from extract_text_from_pdf
            
        Returns:
            List of PDFSection objects representing detected sections
        """
        sections = []
        current_section = None
        current_content = []
        current_page_start = 1
        
        # Iterate through all pages and lines to detect sections
        for page_data in pages:
            page_num = page_data['page_number']
            text = page_data['text']
            lines = text.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Get context lines for header detection
                next_lines = [
                    lines[j].strip() 
                    for j in range(i+1, min(i+5, len(lines)))
                ]
                is_header, level, title = PDFProcessor.is_likely_header(
                    line, next_lines
                )
                
                if is_header and title:
                    # Save previous section if it exists
                    if current_section and current_content:
                        section = PDFProcessor._create_section(
                            current_section,
                            current_content,
                            current_page_start,
                            page_num - 1 if page_num > 1 else page_num
                        )
                        if section:
                            sections.append(section)
                    
                    # Start new section
                    current_section = title
                    current_content = []
                    current_page_start = page_num
                else:
                    # Add line to current section content
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
        """
        Create a PDFSection object from content if it meets minimum requirements.
        
        Args:
            title: Section title
            content: List of content lines
            page_start: Starting page number
            page_end: Ending page number
            
        Returns:
            PDFSection object if valid, None otherwise
        """
        content_text = PDFProcessor.clean_text(' '.join(content))
        word_count = len(content_text.split())
        
        # Check if section meets minimum requirements
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
        """
        Fallback method: divide document by page clusters when section detection fails.
        
        Args:
            pages: List of page dictionaries
            
        Returns:
            List of PDFSection objects created from page clusters
        """
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
            
            # Only create section if it meets minimum requirements
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
        """
        Extract a title from content or generate a default one.
        
        Args:
            content: Section content text
            section_num: Section number for fallback title
            page_group: List of pages in this section
            
        Returns:
            Extracted or generated title string
        """
        sentences = [
            s.strip() 
            for s in content.split('.') 
            if len(s.strip()) > 20
        ]
        
        # Try to find a suitable sentence as title
        if sentences:
            for sentence in sentences[:5]:
                if 10 < len(sentence) < 100 and sentence[0].isupper():
                    return (
                        sentence[:80] + ('...' if len(sentence) > 80 else '')
                    )
        
        # Default title with page range
        return (
            f"Section {section_num}: "
            f"Pages {page_group[0]['page_number']}-{page_group[-1]['page_number']}"
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_section_summary(section: PDFSection) -> Dict[str, Any]:
    """
    Get a summary dictionary of a PDFSection.
    
    Args:
        section: PDFSection object
        
    Returns:
        Dictionary with section summary information
    """
    return {
        "title": section.title,
        "pages": f"{section.page_start}-{section.page_end}",
        "word_count": section.word_count,
        "preview": section.preview
    }


def filter_sections_by_word_count(
    sections: List[PDFSection],
    min_words: int = MIN_WORD_COUNT
) -> List[PDFSection]:
    """
    Filter sections by minimum word count.
    
    Args:
        sections: List of PDFSection objects
        min_words: Minimum word count threshold
        
    Returns:
        Filtered list of PDFSection objects
    """
    return [s for s in sections if s.word_count >= min_words]


def merge_short_sections(
    sections: List[PDFSection],
    min_words: int = MIN_WORD_COUNT
) -> List[PDFSection]:
    """
    Merge consecutive short sections to meet minimum word count.
    
    Args:
        sections: List of PDFSection objects
        min_words: Minimum word count threshold
        
    Returns:
        List of PDFSection objects with merged sections
    """
    if not sections:
        return []
    
    merged = []
    accumulator = None
    
    for section in sections:
        if accumulator is None:
            accumulator = section
        elif accumulator.word_count < min_words:
            # Merge with previous
            accumulator = PDFSection(
                title=f"{accumulator.title} & {section.title}",
                content=f"{accumulator.content}\n\n{section.content}",
                page_start=accumulator.page_start,
                page_end=section.page_end,
                level=min(accumulator.level, section.level),
                word_count=accumulator.word_count + section.word_count,
                preview=accumulator.preview
            )
        else:
            merged.append(accumulator)
            accumulator = section
    
    # Add last section
    if accumulator:
        merged.append(accumulator)
    
    return merged