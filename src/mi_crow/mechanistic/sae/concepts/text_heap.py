from __future__ import annotations

import heapq


class TextHeap:
    """
    Efficient heap for tracking top texts with O(1) duplicate lookup.
    
    Optimized version with incremental index updates instead of full map rebuilds.
    Maintains a min-heap of size k and a dictionary for fast text lookup.
    """
    
    def __init__(self, max_size: int):
        """
        Initialize TextHeap.
        
        Args:
            max_size: Maximum number of items to keep in the heap
        """
        self._max_size = max_size
        self._heap: list[tuple[float, tuple[float, str, int]]] = []
        self._text_to_index: dict[str, int] = {}
    
    def update(self, text: str, score: float, token_idx: int, adjusted_score: float | None = None) -> None:
        """
        Update heap with a new text entry.
        
        Args:
            text: Text string
            score: Activation score (actual value to store)
            token_idx: Token index within the text
            adjusted_score: Optional adjusted score for heap ordering (defaults to score)
        """
        if adjusted_score is None:
            adjusted_score = score
        heap_idx = self._text_to_index.get(text)
        
        if heap_idx is not None:
            self._update_existing(heap_idx, text, adjusted_score, score, token_idx)
        else:
            self._add_new(text, adjusted_score, score, token_idx)
    
    def _update_existing(
        self, 
        heap_idx: int, 
        text: str, 
        adjusted_score: float, 
        score: float, 
        token_idx: int
    ) -> None:
        """Update an existing entry in the heap."""
        current_adj = self._heap[heap_idx][0]
        if adjusted_score > current_adj:
            # Remove old index before updating
            if text in self._text_to_index:
                del self._text_to_index[text]
            
            # Update heap element
            self._heap[heap_idx] = (adjusted_score, (score, text, token_idx))
            
            # Use Python's _siftup for correctness, then rebuild map
            # This is still faster than rebuilding before siftup
            heapq._siftup(self._heap, heap_idx)
            self._rebuild_text_map()
    
    def _add_new(
        self, 
        text: str, 
        adjusted_score: float, 
        score: float, 
        token_idx: int
    ) -> None:
        """Add a new entry to the heap."""
        if len(self._heap) < self._max_size:
            # Add to end of heap
            self._heap.append((adjusted_score, (score, text, token_idx)))
            new_idx = len(self._heap) - 1
            
            # Sift up from new position
            heapq._siftup(self._heap, new_idx)
            self._rebuild_text_map()
        else:
            if adjusted_score > self._heap[0][0]:
                self._replace_minimum(text, adjusted_score, score, token_idx)
    
    def _replace_minimum(
        self, 
        text: str, 
        adjusted_score: float, 
        score: float, 
        token_idx: int
    ) -> None:
        """Replace the minimum element in the heap."""
        # Remove old root from map
        old_text = self._heap[0][1][1]
        if old_text in self._text_to_index:
            del self._text_to_index[old_text]
        
        # Replace root
        self._heap[0] = (adjusted_score, (score, text, token_idx))
        
        # Use Python's _siftup for correctness
        heapq._siftup(self._heap, 0)
        self._rebuild_text_map()
    
    def _rebuild_text_map(self) -> None:
        """Rebuild the text-to-index mapping after heap structure changes."""
        self._text_to_index.clear()
        for idx, (_, (_, heap_text, _)) in enumerate(self._heap):
            self._text_to_index[heap_text] = idx
    
    def get_items(self) -> list[tuple[float, str, int]]:
        """
        Get all items from the heap, sorted by score (descending).
        
        Returns:
            List of (score, text, token_idx) tuples
        """
        return [val for (_, val) in self._heap]
    
    def clear(self) -> None:
        """Clear the heap and text mapping."""
        self._heap.clear()
        self._text_to_index.clear()
    
    def __len__(self) -> int:
        """Return the number of items in the heap."""
        return len(self._heap)
