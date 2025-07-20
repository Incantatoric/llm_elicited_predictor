"""
한화솔루션 LLM-elicited Bayesian stock prediction package
"""

from .data.collector import HanwhaDataCollector
from .elicitation.llm_elicitor import HanwhaLLMElicitor
from .models.bayesian import HanwhaBayesianModel, create_uninformative_model
from .analysis.interpretability import HanwhaInterpreter, BUSINESS_SCENARIOS

__version__ = "0.1.0"
__all__ = [
    "HanwhaDataCollector",
    "HanwhaLLMElicitor", 
    "HanwhaBayesianModel",
    "create_uninformative_model",
    "HanwhaInterpreter",
    "BUSINESS_SCENARIOS"
]
