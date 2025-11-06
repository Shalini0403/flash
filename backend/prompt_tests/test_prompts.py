"""
Minimal Prompt Testing Script
Tests qwen-1.5b model with different prompts and logs results
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to import from main.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import JSONParser, ModelManager
from prompts import PROMPTS


def test_prompts(study_text_path: str, temperature: float = 0.7):
    """
    Test all prompts with qwen-1.5b and log results
    
    Args:
        study_text_path: Path to study text file
        temperature: Temperature for generation (default 0.7)
    """
    
    # Load study text
    print(f"Loading study text from: {study_text_path}")
    with open(study_text_path, 'r', encoding='utf-8') as f:
        study_text = f.read()
    
    print(f"Text length: {len(study_text)} characters\n")
    
    # Setup
    model = "qwen-1.5b"
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test each prompt
    print("="*70)
    print(f"Testing {len(PROMPTS)} prompts with model: {model}")
    print(f"Temperature: {temperature}")
    print("="*70 + "\n")
    
    for prompt_name, prompt_template in PROMPTS.items():
        print(f"\n{'─'*70}")
        print(f"Testing: {prompt_name}")
        print("─"*70)
        
        # Format prompt
        prompt = prompt_template.format(source_text=study_text)
        
        # Generate
        start_time = time.time()
        try:
            raw_response = ModelManager.generate_text(
                model_key=model,
                prompt=prompt,
                temperature=temperature,
                max_new_tokens=2048
            )
            generation_time = (time.time() - start_time) * 1000
            
            print(f"✓ Generated ({generation_time:.0f}ms)")
            print(f"  Response length: {len(raw_response)} chars")
            
            # Parse
            success = False
            parsed_cards = None
            error = None
            
            try:
                data = JSONParser.extract_and_fix_json(raw_response)
                cards = data.get("cards", [])
                parsed_cards = JSONParser.validate_cards(cards)
                
                if parsed_cards:
                    success = True
                    print(f"✓ Parsed {len(parsed_cards)} cards")
                else:
                    error = "No valid cards found"
                    print(f"✗ {error}")
            
            except Exception as e:
                # Try salvage
                salvaged = JSONParser.salvage_partial_cards(raw_response)
                if salvaged:
                    parsed_cards = salvaged
                    success = True
                    print(f"✓ Salvaged {len(salvaged)} cards")
                else:
                    error = f"Parse failed: {str(e)}"
                    print(f"✗ {error}")
            
            # Log result
            result = {
                "prompt_name": prompt_name,
                "model": model,
                "temperature": temperature,
                "success": success,
                "card_count": len(parsed_cards) if parsed_cards else 0,
                "generation_time_ms": generation_time,
                "response_length": len(raw_response),
                "error": error,
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "raw_response": raw_response,
                "cards": parsed_cards if parsed_cards else [],
            }
            results.append(result)
        
        except Exception as e:
            generation_time = (time.time() - start_time) * 1000
            error = f"Generation failed: {str(e)}"
            print(f"✗ {error}")
            
            result = {
                "prompt_name": prompt_name,
                "model": model,
                "temperature": temperature,
                "success": False,
                "card_count": 0,
                "generation_time_ms": generation_time,
                "response_length": 0,
                "error": error,
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "raw_response": "",
                "cards": [],
            }
            results.append(result)
    
    # Save results
    output_file = f"prompt_test_results_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\nTotal prompts tested: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        avg_cards = sum(r['card_count'] for r in successful) / len(successful)
        avg_time = sum(r['generation_time_ms'] for r in successful) / len(successful)
        print(f"\nAverage cards per prompt: {avg_cards:.1f}")
        print(f"Average generation time: {avg_time:.0f}ms")
        
        # Sort by card count
        print("\nPrompts ranked by card count:")
        sorted_results = sorted(successful, key=lambda x: x['card_count'], reverse=True)
        for i, r in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {r['prompt_name']}: {r['card_count']} cards ({r['generation_time_ms']:.0f}ms)")
    
    print(f"\n✓ Full results saved to: {output_file}")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_prompts.py <path_to_study_text> [temperature]")
        print("\nExample:")
        print("  python test_prompts.py ../data/test_texts/biology.txt")
        print("  python test_prompts.py ../data/test_texts/biology.txt 0.5")
        sys.exit(1)
    
    study_text_path = sys.argv[1]
    temperature = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    
    if not Path(study_text_path).exists():
        print(f"Error: File not found: {study_text_path}")
        sys.exit(1)
    
    test_prompts(study_text_path, temperature)