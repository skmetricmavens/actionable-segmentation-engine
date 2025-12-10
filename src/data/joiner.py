"""
Module: joiner

Purpose: Customer ID unification with merge chain resolution.

Implements the critical customer ID merge resolution algorithm that:
- Resolves all merge chains to canonical IDs
- Handles recursive merges (A -> B -> C resolves to A -> C, B -> C)
- Detects circular merges and raises appropriate errors
- Applies merge mapping to event records to prevent double-counting

This is CRITICAL for preventing fragmented customer journeys and inflated metrics.
"""

from typing import Iterator

from src.data.schemas import (
    CustomerIdHistory,
    EventRecord,
)
from src.exceptions import (
    CircularMergeError,
    CustomerMergeError,
    MergeChainTooDeepError,
)


# Type alias for merge mapping
MergeMap = dict[str, str]


def resolve_customer_merges(
    id_history: list[CustomerIdHistory],
    *,
    max_depth: int = 10,
) -> MergeMap:
    """
    Resolve all customer ID merges to final canonical IDs.

    Handles chains: A -> B -> C resolves to A -> C, B -> C

    This function processes the customer ID history table and creates a mapping
    from any historical ID to the final canonical ID. This is essential for
    correctly attributing all events to the current customer identity.

    Args:
        id_history: List of CustomerIdHistory records from customers_id_history table
        max_depth: Maximum merge chain depth (prevents infinite loops)

    Returns:
        Complete mapping of any historical ID to final canonical ID

    Raises:
        CircularMergeError: If a circular merge chain is detected
        MergeChainTooDeepError: If max_depth is exceeded

    Example:
        >>> history = [
        ...     CustomerIdHistory(internal_customer_id="C", past_id="B", ...),
        ...     CustomerIdHistory(internal_customer_id="B", past_id="A", ...),
        ... ]
        >>> merge_map = resolve_customer_merges(history)
        >>> merge_map["A"]  # Returns "C" (A -> B -> C)
        "C"
        >>> merge_map["B"]  # Returns "C"
        "C"
    """
    if not id_history:
        return {}

    # Build initial mapping: past_id -> internal_customer_id
    initial_map: dict[str, str] = {}
    for record in id_history:
        if record.past_id in initial_map:
            # Same past_id merged into different canonical IDs - data quality issue
            raise CustomerMergeError(
                f"Customer ID {record.past_id} appears to be merged into multiple "
                f"canonical IDs: {initial_map[record.past_id]} and {record.internal_customer_id}",
                customer_ids=[record.past_id, initial_map[record.past_id], record.internal_customer_id],
            )
        initial_map[record.past_id] = record.internal_customer_id

    # Resolve chains: follow each past_id until we find the final canonical ID
    canonical_map: MergeMap = {}

    for past_id in initial_map:
        current_id = initial_map[past_id]
        visited: set[str] = {past_id}
        chain_path: list[str] = [past_id]
        depth = 0

        # Follow the chain until we find an ID that's not itself a past_id
        while current_id in initial_map and depth < max_depth:
            if current_id in visited:
                # Circular merge detected
                chain_path.append(current_id)
                raise CircularMergeError(
                    f"Circular merge chain detected: {' -> '.join(chain_path)}",
                    cycle_path=chain_path,
                )

            visited.add(current_id)
            chain_path.append(current_id)
            current_id = initial_map[current_id]
            depth += 1

        if depth >= max_depth:
            chain_path.append(current_id)
            raise MergeChainTooDeepError(
                f"Merge chain exceeded max_depth of {max_depth}: {' -> '.join(chain_path[:5])}...",
                max_depth=max_depth,
                actual_depth=depth,
                chain_path=chain_path,
            )

        # current_id is now the final canonical ID
        canonical_map[past_id] = current_id

    return canonical_map


def apply_merge_to_customer_id(
    customer_id: str,
    merge_map: MergeMap,
) -> str:
    """
    Apply merge mapping to a single customer ID.

    Args:
        customer_id: The customer ID to potentially remap
        merge_map: Mapping from past_id to canonical_id

    Returns:
        The canonical customer ID (original if not in merge_map)
    """
    return merge_map.get(customer_id, customer_id)


def apply_merges_to_events(
    events: list[EventRecord],
    merge_map: MergeMap,
) -> list[EventRecord]:
    """
    Apply merge mapping to all events, returning new EventRecords with remapped IDs.

    This function creates new EventRecord objects with the canonical customer ID
    while preserving all other event properties. Events with IDs not in the merge_map
    are returned unchanged.

    Args:
        events: List of event records
        merge_map: Mapping from past_id to canonical_id

    Returns:
        List of EventRecord objects with remapped customer IDs

    Note:
        Since EventRecord is immutable (frozen=True), new objects are created.
    """
    result: list[EventRecord] = []

    for event in events:
        canonical_id = merge_map.get(event.internal_customer_id)

        if canonical_id is not None:
            # Create new event with remapped customer ID
            remapped_event = EventRecord(
                event_id=event.event_id,
                internal_customer_id=canonical_id,
                event_type=event.event_type,
                timestamp=event.timestamp,
                properties=event.properties,
            )
            result.append(remapped_event)
        else:
            # No mapping needed, keep original
            result.append(event)

    return result


def apply_merges_to_events_iter(
    events: Iterator[EventRecord],
    merge_map: MergeMap,
) -> Iterator[EventRecord]:
    """
    Apply merge mapping to events lazily using an iterator.

    Memory-efficient version for large datasets.

    Args:
        events: Iterator of event records
        merge_map: Mapping from past_id to canonical_id

    Yields:
        EventRecord objects with remapped customer IDs
    """
    for event in events:
        canonical_id = merge_map.get(event.internal_customer_id)

        if canonical_id is not None:
            yield EventRecord(
                event_id=event.event_id,
                internal_customer_id=canonical_id,
                event_type=event.event_type,
                timestamp=event.timestamp,
                properties=event.properties,
            )
        else:
            yield event


def get_all_ids_for_customer(
    canonical_id: str,
    merge_map: MergeMap,
) -> list[str]:
    """
    Get all historical IDs that map to a canonical customer ID.

    Useful for analyzing complete customer journey across ID changes.

    Args:
        canonical_id: The canonical customer ID
        merge_map: Mapping from past_id to canonical_id

    Returns:
        List of all IDs (including canonical) that represent this customer
    """
    all_ids = [canonical_id]

    for past_id, mapped_canonical in merge_map.items():
        if mapped_canonical == canonical_id:
            all_ids.append(past_id)

    return all_ids


def validate_merge_history(
    id_history: list[CustomerIdHistory],
    *,
    max_depth: int = 10,
) -> tuple[bool, list[str]]:
    """
    Validate merge history for potential issues without raising exceptions.

    Useful for data quality checks before processing.

    Args:
        id_history: List of CustomerIdHistory records
        max_depth: Maximum allowed merge chain depth

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues: list[str] = []

    if not id_history:
        return True, []

    # Check for duplicate past_ids
    past_ids = [h.past_id for h in id_history]
    duplicates = [pid for pid in set(past_ids) if past_ids.count(pid) > 1]
    if duplicates:
        issues.append(f"Duplicate past_ids found: {duplicates}")

    # Check for self-references
    for record in id_history:
        if record.past_id == record.internal_customer_id:
            issues.append(f"Self-reference found: {record.past_id}")

    # Try to resolve merges to find circular references or depth issues
    try:
        resolve_customer_merges(id_history, max_depth=max_depth)
    except CircularMergeError as e:
        issues.append(f"Circular merge: {e.message}")
    except MergeChainTooDeepError as e:
        issues.append(f"Chain too deep: {e.message}")
    except CustomerMergeError as e:
        issues.append(f"Merge error: {e.message}")

    return len(issues) == 0, issues


def count_events_by_customer(
    events: list[EventRecord],
) -> dict[str, int]:
    """
    Count events per customer ID.

    Useful for validating merge resolution (total events should be preserved).

    Args:
        events: List of event records

    Returns:
        Dictionary mapping customer_id to event count
    """
    counts: dict[str, int] = {}
    for event in events:
        counts[event.internal_customer_id] = counts.get(event.internal_customer_id, 0) + 1
    return counts


def merge_statistics(
    id_history: list[CustomerIdHistory],
    merge_map: MergeMap,
) -> dict[str, int | float]:
    """
    Calculate statistics about the merge resolution.

    Args:
        id_history: Original merge history
        merge_map: Resolved merge mapping

    Returns:
        Dictionary with merge statistics
    """
    # Count canonical IDs (destinations of merges)
    canonical_ids = set(merge_map.values())

    # Calculate chain depths
    depths: list[int] = []
    initial_map = {h.past_id: h.internal_customer_id for h in id_history}

    for past_id in merge_map:
        depth = 0
        current = initial_map.get(past_id, past_id)
        while current in initial_map and depth < 100:
            current = initial_map[current]
            depth += 1
        depths.append(depth + 1)  # +1 for the initial hop

    return {
        "total_past_ids": len(id_history),
        "unique_canonical_ids": len(canonical_ids),
        "avg_chain_depth": sum(depths) / len(depths) if depths else 0,
        "max_chain_depth": max(depths) if depths else 0,
        "customers_with_merges": len(canonical_ids),
    }
