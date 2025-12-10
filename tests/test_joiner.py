"""
Tests for customer ID merge resolution.
"""

from datetime import datetime, timezone

import pytest

from src.data.joiner import (
    MergeMap,
    apply_merge_to_customer_id,
    apply_merges_to_events,
    apply_merges_to_events_iter,
    count_events_by_customer,
    get_all_ids_for_customer,
    merge_statistics,
    resolve_customer_merges,
    validate_merge_history,
)
from src.data.schemas import (
    CustomerIdHistory,
    EventRecord,
    EventType,
)
from src.exceptions import (
    CircularMergeError,
    CustomerMergeError,
    MergeChainTooDeepError,
)


# =============================================================================
# FIXTURES
# =============================================================================


def make_history(past_id: str, canonical_id: str) -> CustomerIdHistory:
    """Helper to create CustomerIdHistory records."""
    return CustomerIdHistory(
        internal_customer_id=canonical_id,
        past_id=past_id,
        merge_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
    )


def make_event(
    event_id: str,
    customer_id: str,
    event_type: EventType = EventType.SESSION_START,
) -> EventRecord:
    """Helper to create EventRecord."""
    return EventRecord(
        event_id=event_id,
        internal_customer_id=customer_id,
        event_type=event_type,
        timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
    )


# =============================================================================
# BASIC MERGE RESOLUTION TESTS
# =============================================================================


class TestResolveMerges:
    """Tests for resolve_customer_merges function."""

    def test_empty_history(self) -> None:
        """Test with empty history returns empty map."""
        result = resolve_customer_merges([])
        assert result == {}

    def test_single_merge(self) -> None:
        """Test single merge A -> B."""
        history = [make_history("A", "B")]
        result = resolve_customer_merges(history)

        assert result == {"A": "B"}

    def test_multiple_independent_merges(self) -> None:
        """Test multiple independent merges."""
        history = [
            make_history("A", "B"),
            make_history("C", "D"),
            make_history("E", "F"),
        ]
        result = resolve_customer_merges(history)

        assert result == {"A": "B", "C": "D", "E": "F"}

    def test_chain_merge_two_hops(self) -> None:
        """Test chain: A -> B -> C resolves to A -> C, B -> C."""
        history = [
            make_history("A", "B"),
            make_history("B", "C"),
        ]
        result = resolve_customer_merges(history)

        assert result["A"] == "C"  # A resolved through B to C
        assert result["B"] == "C"  # B resolves directly to C

    def test_chain_merge_three_hops(self) -> None:
        """Test chain: A -> B -> C -> D."""
        history = [
            make_history("A", "B"),
            make_history("B", "C"),
            make_history("C", "D"),
        ]
        result = resolve_customer_merges(history)

        assert result["A"] == "D"
        assert result["B"] == "D"
        assert result["C"] == "D"

    def test_multiple_sources_to_same_target(self) -> None:
        """Test multiple IDs merging into same canonical ID."""
        history = [
            make_history("A", "X"),
            make_history("B", "X"),
            make_history("C", "X"),
        ]
        result = resolve_customer_merges(history)

        assert result["A"] == "X"
        assert result["B"] == "X"
        assert result["C"] == "X"

    def test_complex_merge_pattern(self) -> None:
        """Test complex merge: A->B->D, C->D."""
        history = [
            make_history("A", "B"),
            make_history("B", "D"),
            make_history("C", "D"),
        ]
        result = resolve_customer_merges(history)

        assert result["A"] == "D"
        assert result["B"] == "D"
        assert result["C"] == "D"


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestMergeErrors:
    """Tests for merge error handling."""

    def test_circular_merge_simple(self) -> None:
        """Test circular merge A -> B -> A raises error."""
        history = [
            make_history("A", "B"),
            make_history("B", "A"),
        ]

        with pytest.raises(CircularMergeError) as exc_info:
            resolve_customer_merges(history)

        assert "Circular merge" in str(exc_info.value)
        assert exc_info.value.cycle_path is not None

    def test_circular_merge_longer(self) -> None:
        """Test circular merge A -> B -> C -> A."""
        history = [
            make_history("A", "B"),
            make_history("B", "C"),
            make_history("C", "A"),
        ]

        with pytest.raises(CircularMergeError):
            resolve_customer_merges(history)

    def test_max_depth_exceeded(self) -> None:
        """Test chain exceeding max depth raises error."""
        # Create chain of 15 merges
        history = [make_history(f"id_{i}", f"id_{i + 1}") for i in range(15)]

        with pytest.raises(MergeChainTooDeepError) as exc_info:
            resolve_customer_merges(history, max_depth=10)

        assert exc_info.value.max_depth == 10
        assert exc_info.value.actual_depth >= 10

    def test_duplicate_past_id(self) -> None:
        """Test same past_id merged into different canonical IDs raises error."""
        history = [
            make_history("A", "B"),
            make_history("A", "C"),  # Same past_id, different canonical
        ]

        with pytest.raises(CustomerMergeError):
            resolve_customer_merges(history)


# =============================================================================
# APPLY MERGE TESTS
# =============================================================================


class TestApplyMerges:
    """Tests for applying merges to customer IDs and events."""

    def test_apply_to_single_id_with_mapping(self) -> None:
        """Test applying merge to ID in mapping."""
        merge_map: MergeMap = {"A": "B", "C": "D"}
        result = apply_merge_to_customer_id("A", merge_map)
        assert result == "B"

    def test_apply_to_single_id_without_mapping(self) -> None:
        """Test applying merge to ID not in mapping returns original."""
        merge_map: MergeMap = {"A": "B"}
        result = apply_merge_to_customer_id("X", merge_map)
        assert result == "X"

    def test_apply_to_events_empty_map(self) -> None:
        """Test applying empty merge map keeps events unchanged."""
        events = [
            make_event("e1", "cust_1"),
            make_event("e2", "cust_2"),
        ]
        result = apply_merges_to_events(events, {})

        assert len(result) == 2
        assert result[0].internal_customer_id == "cust_1"
        assert result[1].internal_customer_id == "cust_2"

    def test_apply_to_events_with_mapping(self) -> None:
        """Test applying merge map remaps customer IDs."""
        events = [
            make_event("e1", "old_id"),
            make_event("e2", "other_id"),
            make_event("e3", "old_id"),
        ]
        merge_map: MergeMap = {"old_id": "new_id"}

        result = apply_merges_to_events(events, merge_map)

        assert len(result) == 3
        assert result[0].internal_customer_id == "new_id"
        assert result[1].internal_customer_id == "other_id"  # Not in map
        assert result[2].internal_customer_id == "new_id"

    def test_apply_preserves_other_fields(self) -> None:
        """Test merge application preserves event_id, type, timestamp."""
        event = make_event("e1", "old_id", EventType.PURCHASE)
        merge_map: MergeMap = {"old_id": "new_id"}

        result = apply_merges_to_events([event], merge_map)

        assert result[0].event_id == "e1"
        assert result[0].event_type == EventType.PURCHASE
        assert result[0].timestamp == event.timestamp

    def test_apply_events_preserves_count(self) -> None:
        """Test total event count preserved after merge application."""
        events = [
            make_event("e1", "A"),
            make_event("e2", "B"),
            make_event("e3", "A"),
            make_event("e4", "C"),
        ]
        merge_map: MergeMap = {"A": "X", "B": "X"}

        result = apply_merges_to_events(events, merge_map)

        # Count should be same
        assert len(result) == len(events)

        # But customer distribution changes
        counts = count_events_by_customer(result)
        assert counts.get("X") == 3  # A(2) + B(1)
        assert counts.get("C") == 1


class TestApplyMergesIterator:
    """Tests for iterator-based merge application."""

    def test_iterator_produces_same_results(self) -> None:
        """Test iterator version produces same results as list version."""
        events = [
            make_event("e1", "old_id"),
            make_event("e2", "other_id"),
        ]
        merge_map: MergeMap = {"old_id": "new_id"}

        list_result = apply_merges_to_events(events, merge_map)
        iter_result = list(apply_merges_to_events_iter(iter(events), merge_map))

        assert len(list_result) == len(iter_result)
        for lr, ir in zip(list_result, iter_result, strict=True):
            assert lr.event_id == ir.event_id
            assert lr.internal_customer_id == ir.internal_customer_id


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestGetAllIdsForCustomer:
    """Tests for get_all_ids_for_customer function."""

    def test_canonical_only(self) -> None:
        """Test customer with no merges returns only canonical ID."""
        merge_map: MergeMap = {"A": "B", "C": "B"}
        result = get_all_ids_for_customer("X", merge_map)
        assert result == ["X"]

    def test_with_merged_ids(self) -> None:
        """Test customer with merged IDs returns all."""
        merge_map: MergeMap = {"A": "X", "B": "X", "C": "Y"}
        result = get_all_ids_for_customer("X", merge_map)

        assert "X" in result
        assert "A" in result
        assert "B" in result
        assert "C" not in result
        assert len(result) == 3


class TestCountEventsByCustomer:
    """Tests for count_events_by_customer function."""

    def test_count_events(self) -> None:
        """Test counting events per customer."""
        events = [
            make_event("e1", "A"),
            make_event("e2", "A"),
            make_event("e3", "B"),
            make_event("e4", "A"),
        ]
        result = count_events_by_customer(events)

        assert result["A"] == 3
        assert result["B"] == 1

    def test_empty_events(self) -> None:
        """Test empty events returns empty dict."""
        result = count_events_by_customer([])
        assert result == {}


class TestValidateMergeHistory:
    """Tests for validate_merge_history function."""

    def test_valid_history(self) -> None:
        """Test valid history returns True with no issues."""
        history = [
            make_history("A", "B"),
            make_history("C", "D"),
        ]
        is_valid, issues = validate_merge_history(history)

        assert is_valid is True
        assert issues == []

    def test_empty_history_valid(self) -> None:
        """Test empty history is valid."""
        is_valid, issues = validate_merge_history([])
        assert is_valid is True
        assert issues == []

    def test_detects_circular(self) -> None:
        """Test detects circular merge."""
        history = [
            make_history("A", "B"),
            make_history("B", "A"),
        ]
        is_valid, issues = validate_merge_history(history)

        assert is_valid is False
        assert any("Circular" in issue for issue in issues)

    def test_detects_self_reference(self) -> None:
        """Test detects self-reference."""
        history = [
            CustomerIdHistory(
                internal_customer_id="A",
                past_id="A",
                merge_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            )
        ]
        is_valid, issues = validate_merge_history(history)

        assert is_valid is False
        assert any("Self-reference" in issue for issue in issues)


class TestMergeStatistics:
    """Tests for merge_statistics function."""

    def test_statistics_simple(self) -> None:
        """Test statistics with simple merges."""
        history = [
            make_history("A", "X"),
            make_history("B", "X"),
            make_history("C", "Y"),
        ]
        merge_map = resolve_customer_merges(history)
        stats = merge_statistics(history, merge_map)

        assert stats["total_past_ids"] == 3
        assert stats["unique_canonical_ids"] == 2  # X and Y
        assert stats["customers_with_merges"] == 2

    def test_statistics_with_chain(self) -> None:
        """Test statistics with chain merges."""
        history = [
            make_history("A", "B"),
            make_history("B", "C"),
        ]
        merge_map = resolve_customer_merges(history)
        stats = merge_statistics(history, merge_map)

        assert stats["total_past_ids"] == 2
        assert stats["unique_canonical_ids"] == 1  # Only C
        assert stats["max_chain_depth"] >= 2


# =============================================================================
# INTEGRATION TESTS WITH SYNTHETIC DATA
# =============================================================================


class TestWithSyntheticData:
    """Integration tests using synthetic data generator."""

    def test_resolve_synthetic_merges(self) -> None:
        """Test resolving merges from synthetic data."""
        from src.data.synthetic_generator import SyntheticDataGenerator

        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(
            n_customers=100,
            date_range=date_range,
            merge_probability=0.2,
        )

        # Should not raise
        merge_map = resolve_customer_merges(dataset.id_history)

        # All past_ids should be in the map
        for history in dataset.id_history:
            assert history.past_id in merge_map

    def test_apply_merges_preserves_event_count(self) -> None:
        """Test merge application preserves total event count."""
        from src.data.synthetic_generator import SyntheticDataGenerator

        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(
            n_customers=50,
            date_range=date_range,
            merge_probability=0.2,
        )

        merge_map = resolve_customer_merges(dataset.id_history)

        events_before = len(dataset.events)
        events_after = apply_merges_to_events(dataset.events, merge_map)

        # Total event count should be preserved
        assert len(events_after) == events_before

    def test_merged_customers_reduced(self) -> None:
        """Test number of unique customers reduced after merge."""
        from src.data.synthetic_generator import SyntheticDataGenerator

        generator = SyntheticDataGenerator(seed=42)
        date_range = (
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 3, 31, tzinfo=timezone.utc),
        )

        dataset = generator.generate_dataset(
            n_customers=50,
            date_range=date_range,
            merge_probability=0.2,
        )

        if not dataset.id_history:
            pytest.skip("No merges generated in this dataset")

        merge_map = resolve_customer_merges(dataset.id_history)
        merged_events = apply_merges_to_events(dataset.events, merge_map)

        customers_before = len(set(e.internal_customer_id for e in dataset.events))
        customers_after = len(set(e.internal_customer_id for e in merged_events))

        # Should have fewer unique customers after merging
        assert customers_after <= customers_before


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_long_valid_chain(self) -> None:
        """Test long chain within max_depth works."""
        # Create chain of 9 merges (within default max_depth of 10)
        history = [make_history(f"id_{i}", f"id_{i + 1}") for i in range(9)]

        # Should work with max_depth=10
        result = resolve_customer_merges(history, max_depth=10)

        # All should resolve to the final ID
        for i in range(9):
            assert result[f"id_{i}"] == "id_9"

    def test_many_customers_same_canonical(self) -> None:
        """Test many IDs merging to same canonical."""
        history = [make_history(f"old_{i}", "canonical") for i in range(100)]
        result = resolve_customer_merges(history)

        assert len(result) == 100
        assert all(v == "canonical" for v in result.values())

    def test_no_events_to_merge(self) -> None:
        """Test applying merges to empty event list."""
        merge_map: MergeMap = {"A": "B"}
        result = apply_merges_to_events([], merge_map)
        assert result == []
