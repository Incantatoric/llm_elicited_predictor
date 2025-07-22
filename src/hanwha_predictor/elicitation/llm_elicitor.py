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
    Configurable LLM prior elicitation following paper's exact methodology:
    1. Generate variation files (like rephrase_task.py)
    2. Load and use variations (like example_dataset_elicitation.ipynb)
    
    Supports both expert knowledge and data-informed approaches.
    """
    
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo-0125", 
                 include_data_context: bool = False, n_variations: int = 10):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = 0.1  # Paper uses 0.1 for everything
        self.include_data_context = include_data_context
        self.n_variations = n_variations
        
        # Determine approach name and subfolder
        approach_name = "data_informed" if include_data_context else "expert"
        self.approach_name = f"{approach_name}_{n_variations}"
        
        logger.info(f"Initialized HanwhaLLMElicitor with model: {model_name}")
        logger.info(f"Temperature: {self.temperature}")
        logger.info(f"Approach: {self.approach_name}")
        logger.info(f"Include data context: {include_data_context}")
        logger.info(f"Variations per prompt: {n_variations} (total combinations: {n_variations * n_variations})")
    
    def load_training_data(self, data_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Load training data for context preparation (if needed)"""
        if not self.include_data_context:
            return None, None, None
            
        logger.info("Loading training data for data-informed context...")
        
        # Load processed data
        features_df = pd.read_csv(f"{data_dir}/features_standardized.csv", index_col='Date', parse_dates=True)
        targets_df = pd.read_csv(f"{data_dir}/target_returns.csv", index_col='Date', parse_dates=True)
        
        # Load metadata for feature names
        with open(f"{data_dir}/metadata.json", 'r') as f:
            metadata = json.load(f)
        feature_names = metadata['feature_names']
        
        # Create train/test split (same as main evaluation - first 30 for training)
        n_train = 30
        X_train = features_df.iloc[:n_train]
        y_train = targets_df.iloc[:n_train].iloc[:, 0]  # First column is hanwha_stock
        
        logger.info(f"Loaded training data: {X_train.shape[0]} months, {X_train.shape[1]} features")
        logger.info(f"Training period: {X_train.index[0].strftime('%Y-%m-%d')} to {X_train.index[-1].strftime('%Y-%m-%d')}")
        
        return X_train, y_train, feature_names
    
    def prepare_data_context(self, X_train: pd.DataFrame, y_train: pd.Series) -> str:
        """
        Prepare historical data context for data-informed prior elicitation
        Uses full CSV dump with variable explanations like naive LLM approach
        """
        logger.info("Preparing historical data context for data-informed prior elicitation...")
        
        context_lines = []
        
        # Add explanatory header with clear instruction about what we want
        context_lines.append("HISTORICAL DATA FOR REGRESSION COEFFICIENT PRIOR ASSESSMENT:")
        context_lines.append("The following data shows 30 months of historical relationships between")
        context_lines.append("economic variables and Hanwha Solutions stock returns.")
        context_lines.append("You are going to use this data to generate mean and std for each regression coefficient priors in a very specific JSON format(mean = expected correlation strength, std = confidence level).")
        context_lines.append("Later these normal distributions with different means and std will be added to create the prior distribution for the coefficients")
        
        # Variable explanations
        context_lines.append("VARIABLE EXPLANATIONS:")
        context_lines.append("• kospi_return: Monthly % change in Korean KOSPI stock index")
        context_lines.append("• oil_price_change: Monthly % change in crude oil prices")  
        context_lines.append("• usd_krw_change: Monthly % change in USD/KRW exchange rate")
        context_lines.append("• vix_change: Monthly % change in VIX volatility index")
        context_lines.append("• materials_sector_return: Monthly % change in materials sector index")
        context_lines.append("• hanwha_stock: Monthly % return of Hanwha Solutions stock")
        context_lines.append("All variables are standardized (z-score) with mean=0, std=1 for historical period.")
        context_lines.append("Data represents end-of-month values from July 2022 to January 2025.")
        
        # Historical data table
        context_lines.append("HISTORICAL DATA:")
        context_lines.append("Date,KOSPI_Return,Oil_Price_Change,USD_KRW_Change,VIX_Change,Materials_Sector_Return,Hanwha_Stock_Return")
        
        # Add all training data rows
        for i in range(len(X_train)):
            date_str = X_train.index[i].strftime('%Y-%m-%d')
            feature_str = ','.join([f"{val:.4f}" for val in X_train.iloc[i]])
            target_str = f"{y_train.iloc[i]:.4f}"
            context_lines.append(f"{date_str},{feature_str},{target_str}")
        
        context_lines.append("Based on these historical patterns and your own knowledge, provide mean and std for each coefficient prior into the exact JSON format below.")
        context_lines.append("We need COEFFICIENT PRIORS for linear regression, not data generation.")
        
        full_context = "\n".join(context_lines)
        logger.info(f"Prepared data context: {len(full_context)} characters, {len(context_lines)} lines")
    
        return full_context

    def create_base_prompts(self, feature_names: List[str]) -> Tuple[str, str]:
        """Create base prompts using template-based approach to separate fixed vs variable sections"""
        logger.info("Creating base prompts for Hanwha Solutions...")
        
        # Load data context if needed (FIXED SECTION - never rephrased)
        fixed_data_context = ""
        if self.include_data_context:
            X_train, y_train, _ = self.load_training_data()
            if X_train is not None:
                fixed_data_context = self.prepare_data_context(X_train, y_train) + "\n\n"
        
        # FIXED JSON FORMAT (never rephrased)
        fixed_json_format = f"""
REQUIRED OUTPUT FORMAT (use the EXACT format with EXACT feature names):
{{
  "kospi_return": {{"mean": 0.0, "std": 1.0, "reasoning": "Brief explanation"}},
  "oil_price_change": {{"mean": 0.0, "std": 1.0, "reasoning": "Brief explanation"}},
  "usd_krw_change": {{"mean": 0.0, "std": 1.0, "reasoning": "Brief explanation"}},
  "vix_change": {{"mean": 0.0, "std": 1.0, "reasoning": "Brief explanation"}},
  "materials_sector_return": {{"mean": 0.0, "std": 1.0, "reasoning": "Brief explanation"}}
}}
Do NOT create the first-level key for this JSON object. For example, JSON format below is WRONG:
  "regression_coefficient_priors": {{
    "kospi_return": {{"mean": 0.0, "std": 1.0, "reasoning": "Brief explanation"}},
    "oil_price_change": {{"mean": 0.0, "std": 1.0, "reasoning": "Brief explanation"}},
    "usd_krw_change": {{"mean": 0.0, "std": 1.0, "reasoning": "Brief explanation"}},
    "vix_change": {{"mean": 0.0, "std": 1.0, "reasoning": "Brief explanation"}},
    "materials_sector_return": {{"mean": 0.0, "std": 1.0, "reasoning": "Brief explanation"}}
  }}
CRITICAL INSTRUCTIONS:
- Use EXACT feature names above
- mean: expected regression coefficient (how this variable affects stock returns)
- std: your confidence level (smaller = more confident)
- We need REGRESSION COEFFICIENT PRIORS, not data generation
- Respond ONLY with the JSON object
"""
        
        # Base system (VARIABLE SECTION - can be rephrased)
        if self.include_data_context:
            base_system_variable = "You are a Korean stock market expert specializing in Hanwha Solutions. You will analyze historical data patterns to provide expert parameter priors for linear regression models predicting Hanwha Solutions stock returns."
        else:
            base_system_variable = "You are a Korean stock market expert specializing in Hanwha Solutions. You have deep knowledge of how economic factors affect Korean chemical companies. You can provide expert parameter priors for linear regression models."
        
        # Base user instruction (VARIABLE SECTION - can be rephrased)  
        feature_list_str = str(feature_names).replace("'", '"')
        
        if self.include_data_context:
            base_user_variable = f"""Based on the historical patterns shown above, I need your expert assessment of regression coefficient priors for predicting Hanwha Solutions monthly stock returns. My dataset has these standardized features: {feature_list_str}. For each feature, analyze how it affects Hanwha Solutions stock returns and provide your regression coefficient priors."""
        else:
            base_user_variable = f"""I need your expert opinion on regression coefficient priors for predicting Hanwha Solutions monthly stock returns. My dataset has these standardized features: {feature_list_str}. For each feature, analyze how it affects Hanwha Solutions stock returns and provide your regression coefficient priors."""
        
        # Store template parts for later assembly
        self.template_parts = {
            'fixed_data_context': fixed_data_context,
            'fixed_json_format': fixed_json_format,
            'system_variable': base_system_variable,
            'user_variable': base_user_variable
        }
        
        # Assemble complete prompts (for this initial call)
        complete_system = base_system_variable + " You will respond in JSON format."
        complete_user = fixed_data_context + base_user_variable + fixed_json_format
        
        logger.info("Base prompts created with template-based approach")
        logger.info(f"Fixed sections: {len(fixed_data_context) + len(fixed_json_format)} chars")
        logger.info(f"Variable sections: {len(base_system_variable) + len(base_user_variable)} chars")
        
        return complete_system, complete_user
    
    def assemble_prompt_from_template(self, system_variable: str, user_variable: str) -> Tuple[str, str]:
        """Assemble final prompts from fixed and variable template parts"""
        complete_system = system_variable + " You will respond in JSON format."
        
        # Strip whitespace from each part to avoid internal \n\n sequences
        data_part = self.template_parts['fixed_data_context'].strip()
        user_part = user_variable.strip()
        format_part = self.template_parts['fixed_json_format'].strip()
        
        # Join with single newline to create one cohesive prompt
        complete_user = f"{data_part}\n{user_part}\n{format_part}"
        
        return complete_system, complete_user
    
    def task_description_rephrasing(self, base_text: str, n_rephrasings: int = None) -> str:
        """
        Rephrase text while preserving JSON instructions and critical content
        """
        if n_rephrasings is None:
            n_rephrasings = self.n_variations - 1  # Use configurable parameter
            
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
    
    def parse_rephrasings_to_list(self, base_text: str, rephrasings_text: str, n_rephrasings: int = None) -> List[str]:
        """
        Parse the rephrasing output into individual variations
        """
        if n_rephrasings is None:
            n_rephrasings = self.n_variations - 1  # Use configurable parameter
            
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
        target_variations = n_rephrasings + 1
        if len(variations) > target_variations:
            variations = variations[:target_variations]
        elif len(variations) < target_variations:
            # Duplicate some variations to reach target
            while len(variations) < target_variations:
                variations.append(variations[-1])
        
        logger.info(f"Successfully created {len(variations)} variations")
        return variations
    
    def generate_and_save_variation_files(self, feature_names: List[str], 
                                        prompts_dir: str = "data/priors"):
        """
        Step 1: Generate variation files following paper's approach
        """
        logger.info("="*50)
        logger.info("STEP 1: GENERATING VARIATION FILES")
        logger.info(f"Following paper's rephrase_task.py approach")
        logger.info(f"Approach: {self.approach_name}")
        logger.info(f"Variations per prompt: {self.n_variations}")
        logger.info("="*50)
        
        # Create approach-specific prompts directory
        approach_prompts_dir = f"{prompts_dir}/{self.approach_name}"
        Path(approach_prompts_dir).mkdir(parents=True, exist_ok=True)
        
        # Clean up old files to prevent contamination
        system_file = f"{approach_prompts_dir}/system_roles_hanwha.txt"
        user_file = f"{approach_prompts_dir}/user_roles_hanwha.txt"
        metadata_file = f"{approach_prompts_dir}/generation_metadata.json"
        
        for old_file in [system_file, user_file, metadata_file]:
            if Path(old_file).exists():
                Path(old_file).unlink()
                logger.info(f"Cleaned up old file: {old_file}")
        
        # Create base prompts (now creates template parts)
        base_system, base_user = self.create_base_prompts(feature_names)
        
        # Generate system variations (ONLY rephrase the variable part)
        logger.info("Generating system role variations...")
        system_variable_rephrasings = self.task_description_rephrasing(self.template_parts['system_variable'])
        system_variable_variations = self.parse_rephrasings_to_list(
            self.template_parts['system_variable'], 
            system_variable_rephrasings
        )
        
        # Assemble complete system prompts (variable + fixed JSON instruction)
        system_variations = []
        for system_var in system_variable_variations:
            complete_system = system_var + " You will respond in JSON format."
            system_variations.append(complete_system)
        
        # Generate user variations (ONLY rephrase the variable part)
        logger.info("Generating user role variations...")
        user_variable_rephrasings = self.task_description_rephrasing(self.template_parts['user_variable'])
        user_variable_variations = self.parse_rephrasings_to_list(
            self.template_parts['user_variable'], 
            user_variable_rephrasings
        )
        
        # Assemble complete user prompts (fixed data + variable instruction + fixed JSON format)
        user_variations = []
        for user_var in user_variable_variations:
            # Strip whitespace from each part to avoid internal \n\n sequences
            data_part = self.template_parts['fixed_data_context'].strip()
            user_part = user_var.strip()
            format_part = self.template_parts['fixed_json_format'].strip()
            
            # Join with single newline to create one cohesive prompt
            complete_user = f"{data_part}\n{user_part}\n{format_part}"
            user_variations.append(complete_user)
        
        # Ensure exactly n_variations variations each (original + (n_variations-1) rephrased)
        expected_variations = self.n_variations
        if len(system_variations) != expected_variations:
            logger.warning(f"Expected {expected_variations} system variations, got {len(system_variations)}")
        if len(user_variations) != expected_variations:
            logger.warning(f"Expected {expected_variations} user variations, got {len(user_variations)}")
        
        # Save to files (paper's format: separated by \n\n) - OVERWRITE mode
        with open(system_file, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(system_variations))
        
        with open(user_file, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(user_variations))
        
        logger.info(f"Saved {len(system_variations)} system variations to {system_file}")
        logger.info(f"Saved {len(user_variations)} user variations to {user_file}")
        
        # Save metadata
        metadata = {
            'base_system': self.template_parts['system_variable'],
            'base_user': self.template_parts['user_variable'], 
            'system_rephrasings_raw': system_variable_rephrasings,
            'user_rephrasings_raw': user_variable_rephrasings,
            'feature_names': feature_names,
            'n_variations': {'system': len(system_variations), 'user': len(user_variations)},
            'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'approach': self.approach_name,
            'include_data_context': self.include_data_context,
            'target_combinations': self.n_variations * self.n_variations,
            'template_parts': {
                'fixed_data_context_chars': len(self.template_parts['fixed_data_context']),
                'fixed_json_format_chars': len(self.template_parts['fixed_json_format'])
            }
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("="*50)
        logger.info("STEP 1 COMPLETE: Variation files generated")
        logger.info(f"System variations: {len(system_variations)}")
        logger.info(f"User variations: {len(user_variations)}")
        logger.info(f"Target combinations: {self.n_variations * self.n_variations}")
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
    
    def elicit_priors_from_files(self, feature_names: List[str], 
                                prompts_dir: str = "data/priors",
                                save_dir: str = "data/priors") -> np.ndarray:
        """
        Step 2: Load variations and elicit priors (like example_dataset_elicitation.ipynb)
        """
        logger.info("="*50)
        logger.info("STEP 2: ELICITING PRIORS FROM VARIATION FILES")
        logger.info(f"Following paper's example_dataset_elicitation.ipynb")
        logger.info(f"Approach: {self.approach_name}")
        logger.info("="*50)
        
        # Load variations from approach-specific files
        approach_prompts_dir = f"{prompts_dir}/{self.approach_name}"
        system_roles = self.load_prompts(f"{approach_prompts_dir}/system_roles_hanwha.txt")
        user_roles = self.load_prompts(f"{approach_prompts_dir}/user_roles_hanwha.txt")
        
        logger.info(f"Loaded {len(system_roles)} system roles")
        logger.info(f"Loaded {len(user_roles)} user roles")
        logger.info(f"Total combinations: {len(system_roles) * len(user_roles)}")
        
        # Target map for regression (simple)
        target_map = {"return": 0}  # Regression task
        
        # Elicit priors for all combinations
        all_priors = []
        all_reasoning = []
        raw_priors = []  # Store raw JSON responses
        elicitation_mapping = []  # Store prompt mapping info
        elicitation_num = 0
        
        total_combinations = len(system_roles) * len(user_roles)
        logger.info(f"Starting {total_combinations} prior elicitations...")
        
        for system_idx, system_role in enumerate(system_roles):
            for user_idx, user_role in enumerate(user_roles):
                elicitation_num += 1
                
                try:
                    # Elicit prior using paper's temperature
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_role},
                            {"role": "user", "content": user_role}
                        ],
                        temperature=self.temperature,
                        response_format={"type": "json_object"}
                    )
                    
                    # Parse response
                    result_text = response.choices[0].message.content.strip()
                    logger.info(f"{self.approach_name} elicitation {elicitation_num}")
                    logger.info(f"  Raw JSON response: {result_text}")
                    
                    # Store raw response for debugging
                    try:
                        raw_json = json.loads(result_text)
                        raw_priors.append(raw_json)
                    except json.JSONDecodeError:
                        # Handle cleanup if needed
                        cleaned_text = result_text
                        if result_text.startswith("```json"):
                            cleaned_text = result_text[7:]
                        if cleaned_text.endswith("```"):
                            cleaned_text = cleaned_text[:-3]
                        raw_json = json.loads(cleaned_text)
                        raw_priors.append(raw_json)
                    
                    # Process the response using existing logic
                    prior, reasoning = self._process_prior_response(
                        result_text, feature_names, elicitation_num
                    )
                    
                    # DEBUG: Log the prior array right after processing
                    logger.info(f"  🔍 DEBUG: Prior array from _process_prior_response:")
                    logger.info(f"    Shape: {prior.shape}")
                    logger.info(f"    Content: {prior}")
                    logger.info(f"    kospi_return (row 1): {prior[1] if len(prior) > 1 else 'N/A'}")
                    
                    all_priors.append(prior)
                    
                    # DEBUG: Log what was actually added to the list
                    logger.info(f"  🔍 DEBUG: Added to all_priors list:")
                    logger.info(f"    List length now: {len(all_priors)}")
                    logger.info(f"    Last item shape: {all_priors[-1].shape}")
                    logger.info(f"    Last item kospi_return: {all_priors[-1][1] if len(all_priors[-1]) > 1 else 'N/A'}")
                    
                    all_reasoning.append({
                        'elicitation_num': elicitation_num,
                        'system_idx': system_idx,
                        'user_idx': user_idx,
                        'reasoning': reasoning,
                        'approach': self.approach_name
                    })
                    
                    # Store elicitation mapping
                    elicitation_mapping.append({
                        'elicitation_num': elicitation_num,
                        'system_idx': system_idx,
                        'user_idx': user_idx,
                        'system_prompt': system_role,
                        'user_prompt': user_role
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
                        'reasoning': {f: "Failed elicitation" for f in feature_names},
                        'approach': self.approach_name
                    })
                    
                    # Add fallback entries for consistency
                    raw_priors.append({"error": f"Failed elicitation: {str(e)}"})
                    elicitation_mapping.append({
                        'elicitation_num': elicitation_num,
                        'system_idx': system_idx,
                        'user_idx': user_idx,
                        'system_prompt': system_role,
                        'user_prompt': user_role,
                        'error': str(e)
                    })
        
        # Convert to numpy array
        prior_arrays = np.array(all_priors)
        
        # DEBUG: Log the final array before saving
        logger.info("🔍 DEBUG: Final array before saving:")
        logger.info(f"  prior_arrays shape: {prior_arrays.shape}")
        logger.info(f"  prior_arrays[0] (first prior): {prior_arrays[0]}")
        logger.info(f"  prior_arrays[0][1] (first prior, kospi_return): {prior_arrays[0][1] if len(prior_arrays[0]) > 1 else 'N/A'}")
        logger.info(f"  prior_arrays[1] (second prior): {prior_arrays[1]}")
        logger.info(f"  prior_arrays[1][1] (second prior, kospi_return): {prior_arrays[1][1] if len(prior_arrays[1]) > 1 else 'N/A'}")
        
        # Save results in approach-specific subdirectory
        approach_save_dir = f"{save_dir}/{self.approach_name}"
        Path(approach_save_dir).mkdir(parents=True, exist_ok=True)
        
        # Save individual files
        for i, prior_array in enumerate(prior_arrays):
            # DEBUG: Log each array before saving
            logger.info(f"🔍 DEBUG: Saving prior {i}:")
            logger.info(f"  Shape: {prior_array.shape}")
            logger.info(f"  Content: {prior_array}")
            logger.info(f"  kospi_return (row 1): {prior_array[1] if len(prior_array) > 1 else 'N/A'}")
            
            np.save(f"{approach_save_dir}/hanwha_prior_{i}.npy", prior_array)
        
        # Save reasoning data
        with open(f"{approach_save_dir}/reasoning_data.json", 'w') as f:
            json.dump(all_reasoning, f, indent=2)
        
        # Save raw priors (essential for debugging)
        with open(f"{approach_save_dir}/raw_priors.json", 'w') as f:
            json.dump(raw_priors, f, indent=2)
        
        # Save prompt info (essential for analysis)
        # Get base prompts for documentation
        base_system, base_user = self.create_base_prompts(feature_names)
        prompt_info = {
            'base_system': base_system,
            'base_user': base_user,
            'system_variations': system_roles,
            'user_variations': user_roles,
            'elicitation_mapping': elicitation_mapping
        }
        with open(f"{approach_save_dir}/prompt_info.json", 'w') as f:
            json.dump(prompt_info, f, indent=2)
        
        # Save summary
        summary = {
            'n_priors': len(all_priors),
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'array_shape': list(prior_arrays.shape),
            'method': f'{self.approach_name} prior elicitation',
            'temperature': self.temperature,
            'has_reasoning': True,
            'approach': self.approach_name,
            'include_data_context': self.include_data_context,
            'n_variations': self.n_variations,
            'total_combinations': len(all_priors)
        }
        
        with open(f"{approach_save_dir}/summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("="*50)
        logger.info("STEP 2 COMPLETE: Prior elicitation finished")
        logger.info(f"Successfully elicited {len(all_priors)} priors")
        logger.info(f"Reasoning data saved to {approach_save_dir}/reasoning_data.json")
        logger.info(f"Raw priors saved to {approach_save_dir}/raw_priors.json")
        logger.info(f"Prompt info saved to {approach_save_dir}/prompt_info.json")
        logger.info(f"Saved to {approach_save_dir}/")
        logger.info("="*50)
        
        return prior_arrays
    
    def _process_prior_response(self, result_text: str, feature_names: List[str], 
                              elicitation_num: int) -> Tuple[np.ndarray, Dict[str, str]]:
        """
        Process LLM response to extract prior array and reasoning
        (Extracted from elicit_single_prior for reuse)
        """
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
            
            if best_match and highest_similarity > 0.3:
                feature_data = prior_data[best_match]
                
                # Handle both 'std' and 'std_dev' keys
                std_key = 'std' if 'std' in feature_data else ('std_dev' if 'std_dev' in feature_data else None)
                if std_key is None:
                    logger.warning(f"No std/std_dev key found for {feature}, using uninformative prior")
                    feature_priors.append([0.0, 1.0])
                    reasoning_dict[feature] = "No std/std_dev key found"
                    continue
                    
                std_val = max(float(feature_data[std_key]), 1e-3)  # Paper's std clipping
                feature_priors.append([float(feature_data["mean"]), std_val])
                
                # Store reasoning if available
                if "reasoning" in feature_data:
                    reasoning_dict[feature] = feature_data["reasoning"]
                    
                logger.info(f"  {feature}: mean={feature_data['mean']:.3f}, std={feature_data[std_key]:.3f}")
            else:
                logger.warning(f"No match found for {feature}, using uninformative prior")
                feature_priors.append([0.0, 1.0])
                reasoning_dict[feature] = "No expert opinion available"
        
        # Combine into final prior array
        full_prior = [bias_prior] + feature_priors
        prior_array = np.array(full_prior)
        
        return prior_array, reasoning_dict
    
    def run_full_pipeline(self, feature_names: List[str]) -> np.ndarray:
        """
        Run complete two-step pipeline
        """
        logger.info(f"STARTING FULL TWO-STEP PIPELINE - {self.approach_name.upper()}")
        
        # Step 1: Generate variation files
        system_file, user_file = self.generate_and_save_variation_files(feature_names)
        
        # Step 2: Elicit priors from files
        prior_arrays = self.elicit_priors_from_files(feature_names)
        
        logger.info(f"FULL PIPELINE COMPLETE - {self.approach_name.upper()}")
        return prior_arrays
    
    def load_data_from_files(self, data_dir: str = "data/processed") -> Tuple[List[str], Dict]:
        """Load feature names from data files"""
        with open(f"{data_dir}/metadata.json", 'r') as f:
            metadata = json.load(f)
        return metadata['feature_names'], metadata

# Main execution
if __name__ == "__main__":
    import argparse
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Hanwha LLM Prior Elicitation')
    parser.add_argument('--include_data_context', type=str, default='False',
                       choices=['True', 'False', 'true', 'false'],
                       help='Include historical data context (True/False)')
    parser.add_argument('--n_variations', type=int, default=10,
                       help='Number of prompt variations per type (default: 10)')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125',
                       help='OpenAI model to use')
    
    args = parser.parse_args()
    
    # Convert string boolean to actual boolean
    include_data_context = args.include_data_context.lower() == 'true'
    
    # Get API key from environment
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Initialize elicitor with command line arguments
    elicitor = HanwhaLLMElicitor(
        api_key=API_KEY,
        model_name=args.model,
        include_data_context=include_data_context,
        n_variations=args.n_variations
    )
    
    # Load feature names
    feature_names, metadata = elicitor.load_data_from_files()
    
    # Run full pipeline
    prior_arrays = elicitor.run_full_pipeline(feature_names)
    
    # Summary
    print("\n" + "="*60)
    print("HANWHA PRIOR ELICITATION COMPLETE")
    print("="*60)
    print(f"Method: {elicitor.approach_name}")
    print(f"Step 1: Generated variation files")
    print(f"Step 2: Elicited priors from files")
    print(f"Total priors: {len(prior_arrays)}")
    print(f"Features: {feature_names}")
    print(f"Array shape: {prior_arrays.shape}")
    print(f"Temperature: 0.1 (paper exact)")
    print(f"Include data context: {elicitor.include_data_context}")
    print(f"Variations per prompt: {elicitor.n_variations}")
    print(f"Output directory: data/priors/{elicitor.approach_name}/")
    print("="*60)