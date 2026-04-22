"""
MedVigilant Safety Evaluation Layer
====================================

A modular adversarial robustness and safety testing framework
integrated into the MedVigilant medical safety pipeline.

Modules:
    - guardrails: Runtime input/output validators for all AI pipelines
    - adversarial_inputs: Curated library of adversarial test cases
    - evaluator: Core evaluation engine that runs attacks and scores results
"""

from .guardrails import sanitize_input, validate_image, validate_output, confidence_gate
from .adversarial_inputs import ADVERSARIAL_TEST_CASES
from .evaluator import SafetyEvaluator

__all__ = [
    "sanitize_input",
    "validate_image", 
    "validate_output",
    "confidence_gate",
    "ADVERSARIAL_TEST_CASES",
    "SafetyEvaluator",
]
