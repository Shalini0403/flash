"""
PDF Section Extractor - Complete Working Version
Extracts text from PDF and intelligently divides it into chapters/sections/topics
"""

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import PyPDF2


@dataclass
class Section:
    """Represents a section/chapter in the PDF"""
    title: str
    content: str
    page_start: int
    page_end: int
    level: int
    word_count: int


class PDFSectionExtractor:
    """Extract and organize PDF content into major sections"""
    
    CHAPTER_PATTERNS = [
        r'^CHAPTER\s+[IVXLCDM\d]+[\s:.—-]*(.+)$',
        r'^Chapter\s+[IVXLCDM\d]+[\s:.—-]*(.+)$',
        r'^UNIT\s+\d+[\s:.—-]*(.+)$',
        r'^PART\s+[IVXLCDM\d]+[\s:.—-]*(.+)$',
        r'^MODULE\s+\d+[\s:.—-]*(.+)$',
        r'^SECTION\s+[IVXLCDM\d]+[\s:.—-]*(.+)$',
    ]
    
    def __init__(self, min_section_length: int = 2000, min_word_count: int = 300):
        """
        Initialize the extractor
        
        Args:
            min_section_length: Minimum character length (default: 2000)
            min_word_count: Minimum word count (default: 300)
        """
        self.min_section_length = min_section_length
        self.min_word_count = min_word_count
        self.sections: List[Section] = []
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF page by page"""
        pages = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    pages.append({
                        'page_number': page_num + 1,
                        'text': text
                    })
                    
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
        
        return pages
    
    def is_likely_header(self, line: str, next_lines: List[str]) -> tuple:
        """Determine if a line is a major section header"""
        line = line.strip()
        
        if len(line) < 3 or len(line) > 200:
            return False, 0, None
        
        # Check chapter patterns
        for pattern in self.CHAPTER_PATTERNS:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                title = match.group(1).strip() if match.groups() else line
                return True, 1, title
        
        # Check if all caps header
        words = line.split()
        if line.isupper() and 2 <= len(words) <= 15 and 15 <= len(line) <= 120:
            if next_lines and not next_lines[0].strip().isupper():
                return True, 1, line
        
        # Check for numbered chapters
        if re.match(r'^([IVXLCDM]+|\d{1,2})\s+[A-Z][A-Za-z\s]{10,80}$', line):
            return True, 1, line
        
        return False, 0, None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
        return text.strip()
    
    def divide_into_sections(self, pages: List[Dict[str, Any]]) -> List[Section]:
        """Divide PDF into major sections"""
        sections = []
        current_section = None
        current_content = []
        current_page_start = 1
        
        for page_data in pages:
            page_num = page_data['page_number']
            text = page_data['text']
            lines = text.split('\n')
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                if not line:
                    i += 1
                    continue
                
                next_lines = [lines[j].strip() for j in range(i+1, min(i+5, len(lines)))]
                is_header, level, title = self.is_likely_header(line, next_lines)
                
                if is_header and title:
                    # Save previous section
                    if current_section and current_content:
                        content_text = self.clean_text(' '.join(current_content))
                        word_count = len(content_text.split())
                        
                        if (len(content_text) >= self.min_section_length and 
                            word_count >= self.min_word_count):
                            sections.append(Section(
                                title=current_section,
                                content=content_text,
                                page_start=current_page_start,
                                page_end=page_num - 1 if page_num > 1 else page_num,
                                level=1,
                                word_count=word_count
                            ))
                    
                    current_section = title
                    current_content = []
                    current_page_start = page_num
                else:
                    current_content.append(line)
                
                i += 1
        
        # Save last section
        if current_section and current_content:
            content_text = self.clean_text(' '.join(current_content))
            word_count = len(content_text.split())
            
            if (len(content_text) >= self.min_section_length and 
                word_count >= self.min_word_count):
                sections.append(Section(
                    title=current_section,
                    content=content_text,
                    page_start=current_page_start,
                    page_end=pages[-1]['page_number'],
                    level=1,
                    word_count=word_count
                ))
        
        # Fallback if too few sections
        if len(sections) < 2:
            sections = self._fallback_section_division(pages)
        
        return sections
    
    def _fallback_section_division(self, pages: List[Dict[str, Any]]) -> List[Section]:
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
            content = self.clean_text(content)
            word_count = len(content.split())
            
            if len(content) >= self.min_section_length and word_count >= self.min_word_count:
                sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
                title = None
                
                if sentences:
                    for sentence in sentences[:5]:
                        if 10 < len(sentence) < 100 and sentence[0].isupper():
                            title = sentence[:80] + ('...' if len(sentence) > 80 else '')
                            break
                
                if not title:
                    title = f"Section {section_num}: Pages {page_group[0]['page_number']}-{page_group[-1]['page_number']}"
                
                sections.append(Section(
                    title=title,
                    content=content,
                    page_start=page_group[0]['page_number'],
                    page_end=page_group[-1]['page_number'],
                    level=1,
                    word_count=word_count
                ))
                section_num += 1
        
        return sections
    
    def process_pdf(self, pdf_path: str) -> List[Section]:
        """Main method to process PDF"""
        print(f"📄 Processing PDF: {pdf_path}")
        
        pages = self.extract_text_from_pdf(pdf_path)
        print(f"✓ Extracted text from {len(pages)} pages")
        
        sections = self.divide_into_sections(pages)
        print(f"✓ Found {len(sections)} sections")
        
        self.sections = sections
        return sections
    
    def get_sections_summary(self) -> List[Dict[str, Any]]:
        """Get summary without full content"""
        return [
            {
                'title': section.title,
                'page_start': section.page_start,
                'page_end': section.page_end,
                'word_count': section.word_count,
                'level': section.level,
                'preview': section.content[:200] + '...' if len(section.content) > 200 else section.content
            }
            for section in self.sections
        ]
    
    def export_sections_to_dict(self) -> Dict[str, Any]:
        """Export sections as dictionary"""
        return {
            'total_sections': len(self.sections),
            'sections': [asdict(section) for section in self.sections]
        }


def main():
    """Command-line usage"""
    import json
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <path_to_pdf>")
        print("\nExample:")
        print("  python pdf_processor.py textbook.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not Path(pdf_path).exists():
        print(f"❌ Error: File '{pdf_path}' not found")
        sys.exit(1)
    
    extractor = PDFSectionExtractor(min_section_length=2000, min_word_count=300)
    
    try:
        sections = extractor.process_pdf(pdf_path)
        
        print("\n" + "="*80)
        print("📚 SECTIONS FOUND")
        print("="*80)
        
        for i, section in enumerate(sections, 1):
            print(f"\n{i}. {section.title}")
            print(f"   Pages: {section.page_start}-{section.page_end} | Words: {section.word_count}")
            print(f"   Preview: {section.content[:150]}...")
        
        output_file = Path(pdf_path).stem + "_sections.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extractor.export_sections_to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Sections exported to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()