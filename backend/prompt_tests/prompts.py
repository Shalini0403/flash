"""
Prompt Templates for Experimentation
Test different prompting strategies to find optimal approach
"""

# Original baseline prompt from main.py
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

# Variant 1: More structured with examples
STRUCTURED_WITH_EXAMPLES = """
You are an educational flashcard generator. Create high-quality flashcards from study material.

REQUIREMENTS:
1. Generate 8-15 flashcards covering key concepts
2. Questions should be clear and specific
3. Answers should be concise bullet points
4. Include helpful hints when possible

FLASHCARD FORMAT (JSON only):
{{
  "cards": [
    {{
      "front": "What is photosynthesis?",
      "back": ["Process where plants convert light to energy", "Uses CO2 and water", "Produces glucose and oxygen"],
      "hint": "Think about how plants make food"
    }}
  ]
}}

IMPORTANT: Return ONLY valid JSON. No explanations, no markdown, just JSON.

Study Material:
{source_text}
"""

# Variant 2: Emphasis on diversity
DIVERSITY_FOCUSED = """
Generate diverse educational flashcards from the study material below.

Create flashcards with VARIETY:
- Different question types (What, Why, How, When, Who)
- Mix of difficulty levels (basic recall and deeper understanding)
- Cover different topics from the material
- Vary answer lengths (1-4 bullets)

Output Format (JSON only):
{{
  "cards": [
    {{"front": "question", "back": ["answer1", "answer2"], "hint": "optional hint"}}
  ]
}}

Rules:
- Front: Clear question (max 120 chars)
- Back: Array of 1-4 concise points
- Hint: Brief clue (optional)
- Return ONLY JSON

Study Text:
{source_text}
"""

# Variant 3: Minimalist approach
MINIMALIST_PROMPT = """
Create flashcards as JSON from the text below.

Format:
{{
  "cards": [
    {{"front": "question?", "back": ["answer"], "hint": "clue"}}
  ]
}}

Text:
{source_text}
"""

# Variant 4: Step-by-step reasoning
STEP_BY_STEP_PROMPT = """
You are creating educational flashcards. Follow these steps:

Step 1: Read the study material carefully
Step 2: Identify 10-15 key concepts or facts
Step 3: For each concept, create a flashcard with:
   - A clear question (front)
   - 1-4 bullet point answers (back)
   - An optional hint

Step 4: Format as JSON:
{{
  "cards": [
    {{"front": "...", "back": ["...", "..."], "hint": "..."}}
  ]
}}

Return ONLY the JSON object. No explanations.

Study Material:
{source_text}
"""

# Variant 5: Strict JSON schema emphasis
SCHEMA_STRICT_PROMPT = """
Generate flashcards in STRICT JSON format.

JSON SCHEMA:
{{
  "cards": [
    {{
      "front": string (max 120 chars, required),
      "back": array of strings (1-4 items, required),
      "hint": string or null (optional)
    }}
  ]
}}

RULES:
1. MUST be valid JSON
2. MUST include "cards" array
3. Each card MUST have "front" and "back"
4. "hint" is optional
5. NO markdown, NO explanations, NO comments
6. ONLY return the JSON object

Study Text:
{source_text}
"""

# Variant 6: Educational best practices focused
EDUCATIONAL_BEST_PRACTICES = """
You are an expert educator creating high-quality flashcards.

FLASHCARD BEST PRACTICES:
- Ask ONE clear question per card
- Answers should be memorable and concise
- Use active recall techniques
- Include context clues in hints
- Focus on understanding, not just memorization

Create 10-15 flashcards following these principles.

JSON Format:
{{
  "cards": [
    {{"front": "question", "back": ["key point 1", "key point 2"], "hint": "helpful clue"}}
  ]
}}

Output: JSON only, no other text.

Study Material:
{source_text}
"""

# Variant 7: Terse/Direct
TERSE_PROMPT = """
Create flashcard JSON from text.

Format: {{"cards": [{{"front": "Q?", "back": ["A"], "hint": "H"}}]}}

Text:
{source_text}
"""

# Variant 8: Bloom's Taxonomy aligned
BLOOMS_TAXONOMY_PROMPT = """
Generate flashcards at different cognitive levels (Bloom's Taxonomy):

REMEMBER (basic facts):
- "What is...?" 
- "Define..."

UNDERSTAND (explain concepts):
- "Why does...?"
- "Explain how..."

APPLY (use knowledge):
- "How would you use...?"
- "What happens if...?"

Create 12-15 cards across these levels.

JSON Format:
{{
  "cards": [
    {{"front": "question", "back": ["answer"], "hint": "clue"}}
  ]
}}

Return JSON only.

Study Text:
{source_text}
"""

# Variant 9: With explicit card count
EXPLICIT_COUNT_PROMPT = """
Generate EXACTLY 12 flashcards from the study material.

Requirements per card:
- front: One focused question (≤120 chars)
- back: 1-3 concise bullet points
- hint: Optional helpful clue

Output Format (JSON only):
{{
  "cards": [
    {{"front": "...", "back": ["..."], "hint": "..."}},
    ... (12 total)
  ]
}}

Study Material:
{source_text}
"""

# Variant 10: Conversational style
CONVERSATIONAL_PROMPT = """
Hey! I need your help creating flashcards from some study material.

Here's what I need:
- Make them clear and easy to understand
- Questions on the front
- Bullet point answers on the back
- Add hints when they'd be helpful
- Give me about 10-15 cards

Please format as JSON like this:
{{
  "cards": [
    {{"front": "your question", "back": ["point 1", "point 2"], "hint": "your hint"}}
  ]
}}

Just give me the JSON, nothing else.

Here's the material:
{source_text}
"""


# Dictionary of all prompts for easy access - THIS IS WHAT THE TEST FILE IMPORTS
PROMPTS = {
    "baseline": BASELINE_PROMPT,
    "structured_examples": STRUCTURED_WITH_EXAMPLES,
    "diversity_focused": DIVERSITY_FOCUSED,
    "minimalist": MINIMALIST_PROMPT,
    "step_by_step": STEP_BY_STEP_PROMPT,
    "schema_strict": SCHEMA_STRICT_PROMPT,
    "educational_best": EDUCATIONAL_BEST_PRACTICES,
    "terse": TERSE_PROMPT,
    "blooms_taxonomy": BLOOMS_TAXONOMY_PROMPT,
    "explicit_count": EXPLICIT_COUNT_PROMPT,
    "conversational": CONVERSATIONAL_PROMPT,
}

# Backward compatibility alias (in case main.py uses PROMPT_TEMPLATES)
PROMPT_TEMPLATES = PROMPTS


def get_prompt(template_name: str, source_text: str) -> str:
    """
    Get formatted prompt by template name
    
    Args:
        template_name: Name of the template
        source_text: Study text to insert
    
    Returns:
        Formatted prompt string
    """
    if template_name not in PROMPTS:
        raise ValueError(
            f"Unknown template '{template_name}'. "
            f"Available: {list(PROMPTS.keys())}"
        )
    
    return PROMPTS[template_name].format(source_text=source_text)


def list_prompts():
    """List all available prompt templates"""
    return list(PROMPTS.keys())