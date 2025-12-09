"""
Module: exceptions

Purpose: Domain-specific exception hierarchy for the segmentation engine.

Key Classes:
- SegmentationEngineError: Base exception for all engine errors
- DataValidationError: Schema/data validation failures
- CustomerMergeError: Customer ID merge resolution failures
- SegmentRejectedError: Segment fails viability criteria
- SensitivityTestError: Sensitivity analysis failures
- LLMIntegrationError: LLM API/response errors

Architecture Notes:
- All exceptions include context information
- No bare except: clauses allowed
- Exceptions should be raised instead of returning None
"""

# TODO: Implement exception hierarchy
