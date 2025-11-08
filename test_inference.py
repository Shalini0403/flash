"""
Simple Test Script
Reads multiple test cases from a single JSON input file
Tests model with varied prompts and texts
Uses the same functions as main.py - no code rewriting

Directory structure:
project/
  ├── backend/
  │   └── main.py
  ├── test_inference.py (this file)
  └── test_input.json
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

# Import from backend.main (since we're outside backend directory)
from backend.main import AVAILABLE_MODELS, JSONParser, Logger, ModelManager

# Import Groq (install with: pip install groq)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Logger.warning("Groq library not installed. Run: pip install groq")


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "test_input.json"  # JSON file with "test_cases" array
OUTPUT_DIR = "test_results"

# HuggingFace model configuration
HF_MODEL = "qwen-1.5b"  # Change to test different HF models
HF_TEMPERATURE = 0.7

# Groq API configuration
GROQ_MODEL = "llama-3.1-8b-instant"  # Groq model name
GROQ_TEMPERATURE = 0.7
GROQ_API_KEY = None  # Will load from environment variable

# Test with BOTH providers by default
TEST_PROVIDERS = ["huggingface", "groq"]  # Always test both
# Comment out one provider if you want to skip it:
# TEST_PROVIDERS = ["huggingface"]  # Only HF
# TEST_PROVIDERS = ["groq"]  # Only Groq


# ============================================================================
# GROQ API FUNCTIONS
# ============================================================================

def generate_with_groq(prompt: str, model: str, temperature: float) -> str:
    """Generate text using Groq API"""
    if not GROQ_AVAILABLE:
        raise Exception("Groq library not installed")
    
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found in environment variables")
    
    client = Groq(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2048
    )
    
    return response.choices[0].message.content


# ============================================================================
# TEST RUNNER
# ============================================================================

async def run_single_test(test_case: dict, case_number: int, total_cases: int, provider: str):
    """Run a single test case and return results"""
    
    test_name = test_case.get("name", f"test_{case_number}")
    study_text = test_case.get("text", "").strip()
    prompt_template = test_case.get("prompt", "").strip()
    
    Logger.header(f"TEST {case_number}/{total_cases}: {test_name} [{provider.upper()}]")
    
    # Validate test case
    if not study_text:
        Logger.error(f"Test '{test_name}': Empty text field!")
        return None
    
    if not prompt_template:
        Logger.error(f"Test '{test_name}': Empty prompt field!")
        return None
    
    if "{source_text}" not in prompt_template:
        Logger.error(f"Test '{test_name}': Prompt missing {{source_text}} placeholder!")
        return None
    
    Logger.info(f"Provider: {provider}")
    Logger.info(f"Text length: {len(study_text)} characters")
    Logger.info(f"Prompt length: {len(prompt_template)} characters")
    print()
    
    # Generate timestamp for this test
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Select model and temperature based on provider
    if provider == "groq":
        model = GROQ_MODEL
        temperature = GROQ_TEMPERATURE
    else:  # huggingface
        model = HF_MODEL
        temperature = HF_TEMPERATURE
    
    try:
        # Format the prompt with study text
        formatted_prompt = prompt_template.format(source_text=study_text)
        
        # Generate using appropriate provider
        Logger.info(f"Generating with {provider}...")
        if provider == "groq":
            content = generate_with_groq(formatted_prompt, model, temperature)
        else:  # huggingface
            content = ModelManager.generate_text(model, formatted_prompt, temperature)
        
        Logger.success(f"Generation complete ({len(content)} chars)")
        
        # Print raw response to console
        print("\n" + "="*70)
        print(f"RAW OUTPUT ({test_name} - {provider.upper()}):")
        print("="*70)
        print(content)
        print("="*70 + "\n")
        
        # Try to parse the output as JSON (but don't fail if it can't)
        parsed_output = None
        salvaged_cards = None
        parse_status = "failed"
        
        try:
            data = JSONParser.extract_and_fix_json(content)
            cards = data.get("cards", [])
            cards = JSONParser.validate_cards(cards)
            if cards:
                parsed_output = {"cards": cards}
                parse_status = "success"
                Logger.success(f"Successfully parsed {len(cards)} flashcards")
        except Exception as parse_error:
            Logger.warning(f"Full parse failed: {parse_error}")
            # Try to salvage partial cards
            try:
                salvaged = JSONParser.salvage_partial_cards(content)
                if salvaged:
                    salvaged_cards = {"cards": salvaged}
                    parse_status = "partial"
                    Logger.info(f"Salvaged {len(salvaged)} partial cards")
            except Exception as salvage_error:
                Logger.warning(f"Salvage also failed: {salvage_error}")
        
        # Return result dictionary
        result = {
            "test_name": test_name,
            "provider": provider,
            "model": model,
            "timestamp": timestamp,
            "temperature": temperature,
            "parse_status": parse_status,
            "source_text": study_text,
            "prompt_template": prompt_template,
            "formatted_prompt": formatted_prompt,
            "raw_output": content,
            "parsed_output": parsed_output,
            "salvaged_cards": salvaged_cards
        }
        
        Logger.success(f"Test '{test_name}' complete")
        print("\n" + "="*70 + "\n")
        
        return result
        
    except Exception as e:
        Logger.error(f"Test failed: {str(e)}")
        print("\n" + "="*70 + "\n")
        return None


async def run_all_tests():
    """Run all test cases from input file"""
    
    # Read input JSON
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        Logger.error(f"Input file '{INPUT_FILE}' not found!")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            input_data = json.load(f)
        except json.JSONDecodeError as e:
            Logger.error(f"Invalid JSON in input file: {e}")
            return
    
    # Get test cases
    test_cases = input_data.get("test_cases", [])
    
    if not test_cases:
        Logger.error("No test_cases found in input file!")
        Logger.info("Expected format: {\"test_cases\": [{\"name\": \"...\", \"text\": \"...\", \"prompt\": \"...\"}]}")
        return
    
    total_cases = len(test_cases)
    
    # Validate providers
    active_providers = []
    if "huggingface" in TEST_PROVIDERS:
        active_providers.append("huggingface")
    if "groq" in TEST_PROVIDERS:
        if not GROQ_AVAILABLE:
            Logger.error("Groq selected but library not installed. Run: pip install groq")
        elif not (GROQ_API_KEY or os.getenv("GROQ_API_KEY")):
            Logger.error("Groq selected but GROQ_API_KEY not set in environment")
        else:
            active_providers.append("groq")
    
    if not active_providers:
        Logger.error("No valid providers configured!")
        return
    
    Logger.header(f"RUNNING {total_cases} TEST CASES x {len(active_providers)} PROVIDERS")
    Logger.info(f"Providers: {', '.join(active_providers)}")
    print()
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run tests for each provider
    for provider in active_providers:
        Logger.header(f"TESTING WITH: {provider.upper()}")
        print()
        
        provider_results = []
        
        # Run each test case
        for i, test_case in enumerate(test_cases, 1):
            result = await run_single_test(test_case, i, total_cases, provider)
            if result:
                provider_results.append(result)
        
        # Save results for this provider
        # Save consolidated JSON
        json_file = output_dir / f"results_{provider}_{run_timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(provider_results, f, indent=2, ensure_ascii=False)
        
        Logger.success(f"{provider.upper()} results saved to: {json_file}")
        
        # Save consolidated TXT with raw outputs
        txt_file = output_dir / f"raw_outputs_{provider}_{run_timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"ALL RAW OUTPUTS - {provider.upper()} - {run_timestamp}\n")
            if provider == "huggingface":
                f.write(f"Model: {HF_MODEL}, Temperature: {HF_TEMPERATURE}\n")
            else:
                f.write(f"Model: {GROQ_MODEL}, Temperature: {GROQ_TEMPERATURE}\n")
            f.write("="*80 + "\n\n")
            
            for result in provider_results:
                f.write("\n" + "="*80 + "\n")
                f.write(f"TEST: {result['test_name']}\n")
                f.write(f"Provider: {result['provider']}\n")
                f.write(f"Model: {result['model']}\n")
                f.write(f"Status: {result['parse_status']}\n")
                f.write(f"Timestamp: {result['timestamp']}\n")
                f.write("="*80 + "\n")
                f.write("\nPROMPT TEMPLATE:\n")
                f.write("-"*80 + "\n")
                f.write(result['prompt_template'])
                f.write("\n" + "-"*80 + "\n")
                f.write("\nSOURCE TEXT:\n")
                f.write("-"*80 + "\n")
                f.write(result['source_text'])
                f.write("\n" + "-"*80 + "\n")
                f.write("\nRAW OUTPUT:\n")
                f.write("-"*80 + "\n")
                f.write(result['raw_output'])
                f.write("\n" + "-"*80 + "\n\n")
        
        Logger.success(f"{provider.upper()} raw outputs saved to: {txt_file}")
        
        # Print summary for this provider
        success_count = sum(1 for r in provider_results if r['parse_status'] == 'success')
        partial_count = sum(1 for r in provider_results if r['parse_status'] == 'partial')
        failed_count = sum(1 for r in provider_results if r['parse_status'] == 'failed')
        
        print(f"\n{provider.upper()} SUMMARY:")
        print(f"  ✅ Success: {success_count}")
        print(f"  ⚠️  Partial: {partial_count}")
        print(f"  ❌ Failed: {failed_count}")
        print("\n" + "="*80 + "\n")
    
    Logger.header("ALL TESTS COMPLETE")
    Logger.success(f"Tested {total_cases} cases across {len(active_providers)} provider(s)")
    Logger.info(f"Results in: {OUTPUT_DIR}/")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    Logger.header("COMPREHENSIVE INFERENCE TEST")
    Logger.info(f"Reading from: {INPUT_FILE}")
    Logger.info(f"Output to: {OUTPUT_DIR}/")
    
    print("\nCONFIGURATION:")
    if "huggingface" in TEST_PROVIDERS:
        print(f"  HuggingFace: {HF_MODEL} (temp: {HF_TEMPERATURE})")
        print(f"  Available HF models: {AVAILABLE_MODELS}")
    if "groq" in TEST_PROVIDERS:
        print(f"  Groq: {GROQ_MODEL} (temp: {GROQ_TEMPERATURE})")
        groq_key_status = "✓ Set" if (GROQ_API_KEY or os.getenv("GROQ_API_KEY")) else "✗ Not set"
        print(f"  Groq API Key: {groq_key_status}")
    print()
    
    asyncio.run(run_all_tests())