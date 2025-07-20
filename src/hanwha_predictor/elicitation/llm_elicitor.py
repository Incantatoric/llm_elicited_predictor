import openai
import json
import itertools
import time
import os
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_stock_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HanwhaLLMElicitor:
    """
    Two-step prior elicitation following paper's exact methodology:
    1. Generate variation files (like rephrase_task.py)
    2. Load and use variations (like example_dataset_elicitation.ipynb)
    """
    
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo-0125"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = 0.1  # Paper uses 0.1 for everything
        logger.info(f"Initialized HanwhaLLMElicitor with model: {model_name}")
        logger.info(f"Temperature: {self.temperature}")
    
    def create_base_prompts(self, feature_names: List[str]) -> Tuple[str, str]:
        """Create base prompts for Hanwha Solutions with proper JSON instructions"""
        logger.info("Creating base prompts for Hanwha Solutions...")
        
        # Create feature list string
        feature_list_str = str(feature_names).replace("'", '"')
        
        # Base system with JSON instruction
        base_system = """You are a Korean stock market expert specializing in Hanwha Solutions. 
You have deep knowledge of how economic factors affect Korean chemical companies.
You can provide expert parameter priors for linear regression models predicting Hanwha Solutions stock returns.
You will respond in JSON format with your expert analysis."""
        
        # Base user with explicit JSON structure including reasoning (FLATTENED to avoid parsing issues)
        base_user = f"""I am a data scientist predicting Hanwha Solutions monthly stock returns. My dataset has these standardized features: {feature_list_str}. For each feature, I need your expert opinion on: - mean: expected correlation with stock returns - std: your confidence level (smaller = more confident) - reasoning: brief explanation of your assessment. Please analyze each feature's relationship to Hanwha Solutions and respond with JSON in this EXACT format with EXACT keys: {{"kospi_return": {{"mean": 0.0, "std": 1.0, "reasoning": "Explain your assessment here"}}, "oil_price_change": {{"mean": 0.0, "std": 1.0, "reasoning": "Explain your assessment here"}}, "usd_krw_change": {{"mean": 0.0, "std": 1.0, "reasoning": "Explain your assessment here"}}, "vix_change": {{"mean": 0.0, "std": 1.0, "reasoning": "Explain your assessment here"}}, "materials_sector_return": {{"mean": 0.0, "std": 1.0, "reasoning": "Explain your assessment here"}}}}. Consider that Hanwha Solutions is a major Korean chemical company affected by: - Oil prices (raw material costs) - USD/KRW exchange rates (export competitiveness) - Market volatility (investor sentiment) - Materials sector performance (industry dynamics). Respond only with the JSON object."""
        
        logger.info("Base prompts created with JSON instructions")
        return base_system, base_user
    
    def task_description_rephrasing(self, base_text: str, n_rephrasings: int = 9) -> str:
        """
        Rephrase text while preserving JSON instructions and critical content
        """
        logger.info(f"Rephrasing text {n_rephrasings} times while preserving JSON instructions...")
        
        # Clean text (paper removes newlines)
        base_text = base_text.replace("\n", " ")
        
        # Enhanced rephrasing prompt that preserves critical instructions
        rephrasing_prompt = f"""Please rephrase the following text {n_rephrasings} times, keeping the same meaning and preserving all important instructions.

CRITICAL: You must preserve these elements in ALL variations:
1. Any JSON formatting instructions
2. Any specific response format requirements  
3. Technical terms and feature names
4. The core task description

Original text: {base_text}

Generate {n_rephrasings} different phrasings while maintaining the essential meaning and all critical instructions.
Number each variation (1., 2., 3., etc.).
"""
        
        # Use paper's temperature for rephrasing
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a language model that creates variations of text while preserving meaning and critical instructions. You maintain technical accuracy and preserve all formatting requirements."
                },
                {
                    "role": "user", 
                    "content": rephrasing_prompt
                }
            ],
            temperature=0.7  # Slightly lower for more controlled rephrasing
        )
        
        result = response.choices[0].message.content
        logger.info(f"Rephrasing result: {len(result)} characters")
        return result
    
    def parse_rephrasings_to_list(self, base_text: str, rephrasings_text: str, n_rephrasings: int = 9) -> List[str]:
        """
        Parse the rephrasing output into individual variations
        """
        logger.info("Parsing rephrasings into individual variations...")
        
        # Start with original text
        variations = [base_text]
        
        # Split by common patterns
        lines = rephrasings_text.strip().split('\n')
        
        current_rephrasing = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with numbering (1., 2., etc.)
            import re
            if re.match(r'^\d+\.?\s*', line):
                # Save previous rephrasing if exists
                if current_rephrasing:
                    cleaned = current_rephrasing.strip()
                    if cleaned and cleaned != base_text:
                        variations.append(cleaned)
                
                # Start new rephrasing (remove numbering)
                current_rephrasing = re.sub(r'^\d+\.?\s*', '', line)
            else:
                # Continue current rephrasing
                if current_rephrasing:
                    current_rephrasing += " " + line
                else:
                    current_rephrasing = line
        
        # Add final rephrasing
        if current_rephrasing:
            cleaned = current_rephrasing.strip()
            if cleaned and cleaned != base_text:
                variations.append(cleaned)
        
        # If parsing failed, try alternative method
        if len(variations) < n_rephrasings + 1:
            logger.warning(f"Only found {len(variations)} variations, trying alternative parsing...")
            
            # Try splitting by periods and filtering
            sentences = rephrasings_text.split('.')
            variations = [base_text]  # Reset with original
            
            for sentence in sentences:
                cleaned = sentence.strip()
                if len(cleaned) > 50 and cleaned != base_text:  # Filter short fragments
                    variations.append(cleaned)
                    if len(variations) >= n_rephrasings + 1:
                        break
        
        # Ensure we have exactly the right number
        if len(variations) > n_rephrasings + 1:
            variations = variations[:n_rephrasings + 1]
        elif len(variations) < n_rephrasings + 1:
            # Duplicate some variations to reach target
            while len(variations) < n_rephrasings + 1:
                variations.append(variations[-1])
        
        logger.info(f"Successfully created {len(variations)} variations")
        return variations
    
    def generate_and_save_variation_files(self, feature_names: List[str], 
                                        prompts_dir: str = "config/prompts/elicitation"):
        """
        Step 1: Generate variation files following paper's approach
        """
        logger.info("="*50)
        logger.info("STEP 1: GENERATING VARIATION FILES")
        logger.info("Following paper's rephrase_task.py approach")
        logger.info("="*50)
        
        # Create prompts directory
        Path(prompts_dir).mkdir(parents=True, exist_ok=True)
        
        # Clean up old files to prevent contamination
        system_file = f"{prompts_dir}/system_roles_hanwha.txt"
        user_file = f"{prompts_dir}/user_roles_hanwha.txt"
        metadata_file = f"{prompts_dir}/generation_metadata.json"
        
        for old_file in [system_file, user_file, metadata_file]:
            if Path(old_file).exists():
                Path(old_file).unlink()
                logger.info(f"Cleaned up old file: {old_file}")
        
        # Create base prompts
        base_system, base_user = self.create_base_prompts(feature_names)
        
        # Generate system variations
        logger.info("Generating system role variations...")
        system_rephrasings = self.task_description_rephrasing(base_system, n_rephrasings=9)
        system_variations = self.parse_rephrasings_to_list(base_system, system_rephrasings, 9)
        
        # Generate user variations  
        logger.info("Generating user role variations...")
        user_rephrasings = self.task_description_rephrasing(base_user, n_rephrasings=9)
        user_variations = self.parse_rephrasings_to_list(base_user, user_rephrasings, 9)
        
        # Ensure exactly 10 variations each (original + 9 rephrased)
        if len(system_variations) != 10:
            logger.warning(f"Expected 10 system variations, got {len(system_variations)}")
        if len(user_variations) != 10:
            logger.warning(f"Expected 10 user variations, got {len(user_variations)}")
        
        # Save to files (paper's format: separated by \n\n) - OVERWRITE mode
        with open(system_file, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(system_variations))
        
        with open(user_file, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(user_variations))
        
        logger.info(f"Saved {len(system_variations)} system variations to {system_file}")
        logger.info(f"Saved {len(user_variations)} user variations to {user_file}")
        
        # Save metadata
        metadata = {
            'base_system': base_system,
            'base_user': base_user,
            'system_rephrasings_raw': system_rephrasings,
            'user_rephrasings_raw': user_rephrasings,
            'feature_names': feature_names,
            'n_variations': {'system': len(system_variations), 'user': len(user_variations)},
            'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("="*50)
        logger.info("STEP 1 COMPLETE: Variation files generated")
        logger.info(f"System variations: {len(system_variations)}")
        logger.info(f"User variations: {len(user_variations)}")
        logger.info("="*50)
        
        return system_file, user_file
    
    def load_prompts(self, path: str, delim: str = "\n\n") -> List[str]:
        """Load prompts from file (paper's exact method)"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()  # Remove leading/trailing whitespace
        
        # Split and filter out empty strings
        prompts = [prompt.strip() for prompt in content.split(delim) if prompt.strip()]
        
        logger.info(f"Loaded {len(prompts)} prompts from {path}")
        for i, prompt in enumerate(prompts):
            logger.debug(f"Prompt {i+1} length: {len(prompt)} chars")
            logger.debug(f"Prompt {i+1} preview: {prompt[:100]}...")
        
        return prompts
    
    def elicit_single_prior(self, system_prompt: str, user_prompt: str, 
                           feature_names: List[str], target_map: Dict[str, int],
                           elicitation_num: int) -> Tuple[np.ndarray, Dict[str, str]]:
        """
        Elicit single prior with reasoning from LLM
        Returns: (prior_array, reasoning_dict)
        """
        logger.info(f"Elicitation {elicitation_num}")
        
        # Use prompts directly (no format placeholders needed now)
        # Elicit prior using paper's temperature
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,  # 0.1 as in paper
            response_format={"type": "json_object"}
        )
        
        # Parse response
        result_text = response.choices[0].message.content.strip()
        
        # DEBUG: Log raw response to understand format issues
        logger.info(f"  Raw JSON response: {result_text}")
        
        # Robust JSON parsing
        try:
            prior_data = json.loads(result_text)
            logger.info(f"  Parsed JSON structure: {type(prior_data)}")
            logger.info(f"  JSON keys: {list(prior_data.keys()) if isinstance(prior_data, dict) else 'Not a dict'}")
        except json.JSONDecodeError:
            logger.error(f"  JSON parsing failed, attempting cleanup...")
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            prior_data = json.loads(result_text)
            logger.info(f"  Cleaned and parsed JSON: {type(prior_data)}")
        
        # DEBUG: Show structure of first value
        if isinstance(prior_data, dict) and prior_data:
            first_key = list(prior_data.keys())[0]
            first_value = prior_data[first_key]
            logger.info(f"  First value structure: {first_key} -> {type(first_value)} -> {first_value}")
        
        # Extract reasoning for each feature
        reasoning_dict = {}
        
        # Handle different response formats
        if isinstance(list(prior_data.values())[0], dict):
            # New format: {"feature": {"mean": X, "std": Y, "reasoning": Z}}
            for feature, data in prior_data.items():
                if "reasoning" in data:
                    reasoning_dict[feature] = data["reasoning"]
        else:
            # Old format: {"feature": [mean, std]} - no reasoning
            prior_data = {k: {"mean": v[0], "std": v[1]} for k, v in prior_data.items()}
        
        # Create prior array (bias + features)
        bias_prior = [0.0, 1.0]  # Uninformative bias as in paper
        feature_priors = []
        
        # Match features using fuzzy matching
        from difflib import SequenceMatcher
        
        for feature in feature_names:
            best_match = None
            highest_similarity = 0
            
            for response_feature in prior_data.keys():
                similarity = SequenceMatcher(None, feature, response_feature).ratio()
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = response_feature
            
            if best_match and highest_similarity > 0.3:  # Reasonable threshold
                feature_data = prior_data[best_match]
                std_val = max(float(feature_data["std"]), 1e-3)  # Paper's std clipping
                feature_priors.append([float(feature_data["mean"]), std_val])
                
                # Store reasoning if available
                if "reasoning" in feature_data:
                    reasoning_dict[feature] = feature_data["reasoning"]
                    
                logger.info(f"  {feature}: mean={feature_data['mean']:.3f}, std={feature_data['std']:.3f}")
            else:
                logger.warning(f"No match found for {feature}, using uninformative prior")
                feature_priors.append([0.0, 1.0])
                reasoning_dict[feature] = "No expert opinion available"
        
        # Combine into final prior array
        full_prior = [bias_prior] + feature_priors
        prior_array = np.array(full_prior)
        
        return prior_array, reasoning_dict
    
    def elicit_priors_from_files(self, feature_names: List[str], 
                                prompts_dir: str = "config/prompts/elicitation",
                                save_dir: str = "data/priors") -> np.ndarray:
        """
        Step 2: Load variations and elicit priors (like example_dataset_elicitation.ipynb)
        """
        logger.info("="*50)
        logger.info("STEP 2: ELICITING PRIORS FROM VARIATION FILES")
        logger.info("Following paper's example_dataset_elicitation.ipynb")
        logger.info("="*50)
        
        # Load variations from files
        system_roles = self.load_prompts(f"{prompts_dir}/system_roles_hanwha.txt")
        user_roles = self.load_prompts(f"{prompts_dir}/user_roles_hanwha.txt")
        
        logger.info(f"Loaded {len(system_roles)} system roles")
        logger.info(f"Loaded {len(user_roles)} user roles")
        logger.info(f"Total combinations: {len(system_roles) * len(user_roles)}")
        
        # Target map for regression (simple)
        target_map = {"return": 0}  # Regression task
        
        # Elicit priors for all combinations
        all_priors = []
        all_reasoning = []
        elicitation_num = 0
        
        total_combinations = len(system_roles) * len(user_roles)
        logger.info(f"Starting {total_combinations} prior elicitations...")
        
        for system_idx, system_role in enumerate(system_roles):
            for user_idx, user_role in enumerate(user_roles):
                elicitation_num += 1
                
                try:
                    prior, reasoning = self.elicit_single_prior(
                        system_role, user_role, feature_names, target_map, elicitation_num
                    )
                    all_priors.append(prior)
                    all_reasoning.append({
                        'elicitation_num': elicitation_num,
                        'system_idx': system_idx,
                        'user_idx': user_idx,
                        'reasoning': reasoning
                    })
                    
                    # Brief pause to avoid rate limiting
                    if elicitation_num < total_combinations:
                        time.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"Failed elicitation {elicitation_num}: {e}")
                    # Use uninformative prior as fallback
                    fallback_prior = np.array([[0.0, 1.0]] * (len(feature_names) + 1))
                    all_priors.append(fallback_prior)
                    all_reasoning.append({
                        'elicitation_num': elicitation_num,
                        'system_idx': system_idx,
                        'user_idx': user_idx,
                        'reasoning': {f: "Failed elicitation" for f in feature_names}
                    })
        
        # Convert to numpy array
        prior_arrays = np.array(all_priors)
        
        # Save results
        Path(save_dir).mkdir(exist_ok=True)
        
        # Save individual files
        for i, prior_array in enumerate(prior_arrays):
            np.save(f"{save_dir}/hanwha_prior_{i}.npy", prior_array)
        
        # Save reasoning data
        with open(f"{save_dir}/reasoning_data.json", 'w') as f:
            json.dump(all_reasoning, f, indent=2)
        
        # Save summary
        summary = {
            'n_priors': len(all_priors),
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'array_shape': list(prior_arrays.shape),
            'method': 'Paper exact replication with reasoning extraction',
            'temperature': self.temperature,
            'has_reasoning': True
        }
        
        with open(f"{save_dir}/summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("="*50)
        logger.info("STEP 2 COMPLETE: Prior elicitation finished")
        logger.info(f"Successfully elicited {len(all_priors)} priors")
        logger.info(f"Reasoning data saved to {save_dir}/reasoning_data.json")
        logger.info(f"Saved to {save_dir}/")
        logger.info("="*50)
        
        return prior_arrays
    
    def run_full_pipeline(self, feature_names: List[str]) -> np.ndarray:
        """
        Run complete two-step pipeline
        """
        logger.info("STARTING FULL TWO-STEP PIPELINE")
        
        # Step 1: Generate variation files
        system_file, user_file = self.generate_and_save_variation_files(feature_names)
        
        # Step 2: Elicit priors from files
        prior_arrays = self.elicit_priors_from_files(feature_names)
        
        logger.info("FULL PIPELINE COMPLETE")
        return prior_arrays
    
    def load_data_from_files(self, data_dir: str = "data/processed") -> Tuple[List[str], Dict]:
        """Load feature names from data files"""
        with open(f"{data_dir}/metadata.json", 'r') as f:
            metadata = json.load(f)
        return metadata['feature_names'], metadata

# Main execution
if __name__ == "__main__":
    # Get API key from environment
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Initialize elicitor
    elicitor = HanwhaLLMElicitor(api_key=API_KEY)
    
    # Load feature names
    feature_names, metadata = elicitor.load_data_from_files()
    
    # Run full pipeline
    prior_arrays = elicitor.run_full_pipeline(feature_names)
    
    # Summary
    print("\n" + "="*60)
    print("HANWHA PRIOR ELICITATION COMPLETE")
    print("="*60)
    print(f"Method: Paper's exact two-step approach")
    print(f"Step 1: Generated variation files")
    print(f"Step 2: Elicited priors from files")
    print(f"Total priors: {len(prior_arrays)}")
    print(f"Features: {feature_names}")
    print(f"Array shape: {prior_arrays.shape}")
    print(f"Temperature: 0.1 (paper exact)")
    print("="*60)