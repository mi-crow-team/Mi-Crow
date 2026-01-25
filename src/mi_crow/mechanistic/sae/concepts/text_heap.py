from __future__ import annotations

import heapq


class TextHeap:
    """
    Efficient heap for tracking top texts with O(1) duplicate lookup.
    
    Optimized with incremental index updates and correct heap operations.
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
            self._heap[heap_idx] = (adjusted_score, (score, text, token_idx))
            self._text_to_index[text] = heap_idx
            self._siftdown_with_tracking(heap_idx)
    
    def _add_new(
        self, 
        text: str, 
        adjusted_score: float, 
        score: float, 
        token_idx: int
    ) -> None:
        """Add a new entry to the heap."""
        if len(self._heap) < self._max_size:
            self._heap.append((adjusted_score, (score, text, token_idx)))
            new_idx = len(self._heap) - 1
            self._text_to_index[text] = new_idx
            self._siftup_with_tracking(new_idx)
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
        old_text = self._heap[0][1][1]
        if old_text in self._text_to_index:
            del self._text_to_index[old_text]
        
        self._heap[0] = (adjusted_score, (score, text, token_idx))
        self._text_to_index[text] = 0
        self._siftdown_with_tracking(0)
    
    def _siftup_with_tracking(self, pos: int) -> None:
        """
        Sift element up in heap (toward root) and update text-to-index map incrementally.
        
        Used when value decreases - compares with parent and moves up.
        Only updates indices that actually change during the sift operation.
        """
        startpos = pos
        newitem = self._heap[pos]
        newitem_text = newitem[1][1]
        
        while pos > 0:
            parentpos = (pos - 1) >> 1
            parent = self._heap[parentpos]
            if newitem[0] >= parent[0]:
                break
            parent_text = parent[1][1]
            self._heap[pos] = parent
            self._text_to_index[parent_text] = pos
            pos = parentpos
        
        self._heap[pos] = newitem
        if pos != startpos:
            self._text_to_index[newitem_text] = pos
    
    def _siftdown_with_tracking(self, pos: int) -> None:
        """
        Sift element down in heap and update text-to-index map incrementally.
        
        Only updates indices that actually change during the sift operation.
        """
        endpos = len(self._heap)
        startpos = pos
        newitem = self._heap[pos]
        newitem_text = newitem[1][1]
        
        childpos = 2 * pos + 1
        while childpos < endpos:
            rightpos = childpos + 1
            if rightpos < endpos and self._heap[rightpos][0] < self._heap[childpos][0]:
                childpos = rightpos
            if newitem[0] < self._heap[childpos][0]:
                break
            child_text = self._heap[childpos][1][1]
            self._heap[pos] = self._heap[childpos]
            self._text_to_index[child_text] = pos
            pos = childpos
            childpos = 2 * pos + 1
        
        self._heap[pos] = newitem
        if pos != startpos:
            self._text_to_index[newitem_text] = pos
    
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
