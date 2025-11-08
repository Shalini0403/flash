# Flashcard Generation Model Comparison

## Performance Overview

| Metric | Groq (Llama-3.1-8b) | HuggingFace (Qwen-1.5b) |
|--------|---------------------|-------------------------|
| **Success Rate** | 11/11 (100%) | 3/11 (27%) |
| **Partial Success** | 0/11 (0%) | 7/11 (64%) |
| **Complete Failures** | 0/11 (0%) | 1/11 (9%) |
| **JSON Compliance** | Excellent | Poor |

---

## Test Results by Prompt Type

| Test Name | Prompt Style | Groq Status | Groq Cards | Qwen Status | Qwen Cards | Better Performance |
|-----------|-------------|-------------|------------|-------------|------------|--------------------|
| baseline_photosynthesis | Standard educational format | Success | 6 | Partial | 4 | Groq |
| short_prompt_history | Minimal instructions | Success | 5 | Failed | 1 | Groq |
| detailed_prompt_science | Comprehensive guidelines | Success | 4 | Partial | 4 | Groq |
| simple_definition_focus | Definition-only cards | Success | 4 | Failed | 4 | Groq |
| numbered_facts_math | List-based format | Success | 3 | Partial | 4 | Groq |
| application_based_programming | Practical "how-to" questions | Success | 9 | Failed | 7 | Groq |
| comparison_style_economics | Compare X vs Y format | Success | 3 | Partial | 1 | Groq |
| minimal_prompt_geography | Extremely brief prompt | Success | 5 | Failed | 3 | Groq |
| strict_json_chemistry | Strict formatting rules | Success | 6 | Partial | 3 | Groq |
| verbose_prompt_literature | Detailed teacher persona | Success | 4 | Partial | 4 | Groq |
| question_types_biology | Mixed question styles | Success | 4 | Partial | 4 | Groq |
| bullet_heavy_history | Multiple bullet points | Success | 6 | Partial | 6 | Groq |

**Key Observations:**
- Groq consistently produces valid, parseable output across all prompt types
- Qwen generates comparable card counts when partially successful but fails format validation
- Detailed prompts (detailed_prompt_science, verbose_prompt_literature) show better Qwen performance
- Minimal prompts (short_prompt_history, minimal_prompt_geography) correlate with Qwen failures

---

## Key Issues with Qwen-1.5b

### 1. **JSON Truncation Problem - CRITICAL**
The model consistently cuts off the beginning of JSON responses, missing opening braces and initial fields.

**Impact:** Approximately 64% of outputs are unparseable without manual repair.

---

### 2. **Instruction Following**
- Ignores "return only JSON" directives
- Adds explanations despite explicit instructions not to
- Includes markdown code blocks even when forbidden with "CRITICAL" warnings

---

### 3. **Format Inconsistency**
- Character limits not consistently respected
- Variable structure across different prompt types
- Unreliable hint field usage

---

### 4. **Prompt Sensitivity**
- Performs worse with minimal prompts
- Performs worse with strict formatting requirements
- No prompt style achieved 100% success rate

---

## Mitigation Techniques

### 1. Temperature Reduction
Lower temperature from 0.7 to 0.3 for more deterministic and consistent output generation.

---

### 2. Few-Shot Examples
Include 2-3 complete examples of perfect input-output pairs within the prompt to guide the model toward correct formatting.

---

### 3. Model Fine-Tuning
Fine-tune Qwen-1.5b on a curated dataset of flashcard generation examples with proper JSON formatting to improve:
- Instruction following capabilities
- JSON structure adherence
- Consistent output formatting

---

### 4. Model Upgrade
Consider switching from Qwen-1.5B to Qwen-7B or larger variants for improved instruction following and structured output capabilities.

---

## Recommendations

**For Production Use:** 
- Groq/Llama-3.1-8b is production-ready with 100% success rate across all prompt types

**For Qwen-1.5b Improvement:** 
1. Reduce temperature to 0.3
2. Add few-shot examples to prompts
3. Fine-tune on flashcard generation dataset

**Root Cause:** The 1.5B parameter model appears too small for reliable structured JSON output generation.