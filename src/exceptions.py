"""
Module: exceptions

Purpose: Domain-specific exception hierarchy for the segmentation engine.

All exceptions include context information and should be raised instead of returning None.
"""

from typing import Any


class SegmentationEngineError(Exception):
    """Base exception for all segmentation engine errors."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        self.message = message
        self.context = context or {}
        super().__init__(self.message)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, context={self.context!r})"


class DataValidationError(SegmentationEngineError):
    """Raised when data fails schema or business validation."""

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        value: Any = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if field is not None:
            ctx["field"] = field
        if value is not None:
            ctx["value"] = value
        super().__init__(message, context=ctx)
        self.field = field
        self.value = value


class CustomerMergeError(SegmentationEngineError):
    """Raised when customer ID merge resolution fails."""

    def __init__(
        self,
        message: str,
        *,
        customer_ids: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if customer_ids is not None:
            ctx["customer_ids"] = customer_ids
        super().__init__(message, context=ctx)
        self.customer_ids = customer_ids or []


class CircularMergeError(CustomerMergeError):
    """Raised when a circular merge chain is detected."""

    def __init__(
        self,
        message: str,
        *,
        cycle_path: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if cycle_path is not None:
            ctx["cycle_path"] = cycle_path
        super().__init__(message, customer_ids=cycle_path, context=ctx)
        self.cycle_path = cycle_path or []


class MergeChainTooDeepError(CustomerMergeError):
    """Raised when merge chain exceeds maximum allowed depth."""

    def __init__(
        self,
        message: str,
        *,
        max_depth: int,
        actual_depth: int,
        chain_path: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        ctx["max_depth"] = max_depth
        ctx["actual_depth"] = actual_depth
        if chain_path is not None:
            ctx["chain_path"] = chain_path
        super().__init__(message, customer_ids=chain_path, context=ctx)
        self.max_depth = max_depth
        self.actual_depth = actual_depth
        self.chain_path = chain_path or []


class SegmentRejectedError(SegmentationEngineError):
    """Raised when a segment fails viability criteria."""

    def __init__(
        self,
        message: str,
        *,
        segment_id: str,
        rejection_reasons: list[str],
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        ctx["segment_id"] = segment_id
        ctx["rejection_reasons"] = rejection_reasons
        super().__init__(message, context=ctx)
        self.segment_id = segment_id
        self.rejection_reasons = rejection_reasons


class SensitivityTestError(SegmentationEngineError):
    """Raised when sensitivity analysis encounters an error."""

    def __init__(
        self,
        message: str,
        *,
        test_type: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        ctx["test_type"] = test_type
        super().__init__(message, context=ctx)
        self.test_type = test_type


class FeatureSensitivityError(SensitivityTestError):
    """Raised when feature sensitivity test fails."""

    def __init__(
        self,
        message: str,
        *,
        dropped_features: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if dropped_features is not None:
            ctx["dropped_features"] = dropped_features
        super().__init__(message, test_type="feature_sensitivity", context=ctx)
        self.dropped_features = dropped_features or []


class TimeWindowSensitivityError(SensitivityTestError):
    """Raised when time window sensitivity test fails."""

    def __init__(
        self,
        message: str,
        *,
        windows_tested: list[int] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if windows_tested is not None:
            ctx["windows_tested"] = windows_tested
        super().__init__(message, test_type="time_window_sensitivity", context=ctx)
        self.windows_tested = windows_tested or []


class LLMIntegrationError(SegmentationEngineError):
    """Raised when LLM API or response processing fails."""

    def __init__(
        self,
        message: str,
        *,
        llm_model: str | None = None,
        prompt_type: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if llm_model is not None:
            ctx["llm_model"] = llm_model
        if prompt_type is not None:
            ctx["prompt_type"] = prompt_type
        super().__init__(message, context=ctx)
        self.llm_model = llm_model
        self.prompt_type = prompt_type


class LLMResponseParseError(LLMIntegrationError):
    """Raised when LLM response cannot be parsed."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str | None = None,
        expected_format: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if raw_response is not None:
            ctx["raw_response"] = raw_response[:500]  # Truncate for logging
        if expected_format is not None:
            ctx["expected_format"] = expected_format
        super().__init__(message, context=ctx)
        self.raw_response = raw_response
        self.expected_format = expected_format


class ClusteringError(SegmentationEngineError):
    """Raised when ML clustering encounters an error."""

    def __init__(
        self,
        message: str,
        *,
        n_clusters: int | None = None,
        n_samples: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if n_clusters is not None:
            ctx["n_clusters"] = n_clusters
        if n_samples is not None:
            ctx["n_samples"] = n_samples
        super().__init__(message, context=ctx)
        self.n_clusters = n_clusters
        self.n_samples = n_samples


class InsufficientDataError(SegmentationEngineError):
    """Raised when there is insufficient data for analysis."""

    def __init__(
        self,
        message: str,
        *,
        required: int,
        actual: int,
        data_type: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        ctx["required"] = required
        ctx["actual"] = actual
        ctx["data_type"] = data_type
        super().__init__(message, context=ctx)
        self.required = required
        self.actual = actual
        self.data_type = data_type


class ProfileBuildError(SegmentationEngineError):
    """Raised when customer profile building fails."""

    def __init__(
        self,
        message: str,
        *,
        customer_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if customer_id is not None:
            ctx["customer_id"] = customer_id
        super().__init__(message, context=ctx)
        self.customer_id = customer_id


class TraitExtractionError(SegmentationEngineError):
    """Raised when trait extraction fails."""

    def __init__(
        self,
        message: str,
        *,
        trait_name: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if trait_name is not None:
            ctx["trait_name"] = trait_name
        super().__init__(message, context=ctx)
        self.trait_name = trait_name


class ReportGenerationError(SegmentationEngineError):
    """Raised when report generation fails."""

    def __init__(
        self,
        message: str,
        *,
        report_type: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if report_type is not None:
            ctx["report_type"] = report_type
        super().__init__(message, context=ctx)
        self.report_type = report_type


class LLMError(LLMIntegrationError):
    """Raised when LLM operations fail (alias for LLMIntegrationError with segment context)."""

    def __init__(
        self,
        message: str,
        *,
        prompt_type: str | None = None,
        segment_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if segment_id is not None:
            ctx["segment_id"] = segment_id
        super().__init__(message, prompt_type=prompt_type, context=ctx)
        self.segment_id = segment_id
