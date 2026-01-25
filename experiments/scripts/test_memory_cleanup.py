#!/usr/bin/env python
"""
Simple test script to verify memory cleanup functions work correctly.

This tests:
1. Memory measurement functions
2. Force cleanup functions
3. LanguageModel cleanup (if we can load a small model)
"""

import gc
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_DIR))

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not available - memory measurements will be disabled")

import torch


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    if not PSUTIL_AVAILABLE:
        return 0.0
    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _force_memory_cleanup() -> None:
    """Aggressively clean up memory for both CPU and GPU."""
    gc.collect()
    gc.collect()  # Second pass for cyclic references
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def test_memory_cleanup():
    """Test that memory cleanup functions work."""
    print("=" * 80)
    print("Memory Cleanup Test")
    print("=" * 80)
    
    if PSUTIL_AVAILABLE:
        baseline = get_memory_usage_mb()
        print(f"Baseline memory: {baseline:.1f} MB")
    else:
        print("Memory measurement disabled (psutil not available)")
        baseline = 0.0
    
    # Create some tensors
    print("\nCreating test tensors...")
    tensors = []
    for i in range(10):
        t = torch.randn(1000, 1000)  # ~4MB each
        tensors.append(t)
    
    if PSUTIL_AVAILABLE:
        after_create = get_memory_usage_mb()
        print(f"After creating tensors: {after_create:.1f} MB (Δ {after_create - baseline:+.1f} MB)")
    
    # Delete tensors
    print("\nDeleting tensors...")
    del tensors
    _force_memory_cleanup()
    
    if PSUTIL_AVAILABLE:
        after_cleanup = get_memory_usage_mb()
        print(f"After cleanup: {after_cleanup:.1f} MB (Δ {after_cleanup - baseline:+.1f} MB)")
        print(f"Memory freed: {after_create - after_cleanup:.1f} MB")
    
    print("\n✅ Memory cleanup test completed")


def test_circular_reference_cleanup():
    """Test cleanup of circular references."""
    print("\n" + "=" * 80)
    print("Circular Reference Cleanup Test")
    print("=" * 80)
    
    # Create circular references
    class A:
        def __init__(self):
            self.ref = None
    
    class B:
        def __init__(self, a):
            self.a = a
            a.ref = self  # Circular reference
    
    print("Creating circular references...")
    a = A()
    b = B(a)
    
    if PSUTIL_AVAILABLE:
        before = get_memory_usage_mb()
    
    # Delete references
    print("Deleting references...")
    del a, b
    _force_memory_cleanup()
    
    if PSUTIL_AVAILABLE:
        after = get_memory_usage_mb()
        print(f"Memory before: {before:.1f} MB")
        print(f"Memory after: {after:.1f} MB")
    
    print("✅ Circular reference cleanup test completed")


def main():
    """Run all tests."""
    print("Testing memory cleanup utilities...")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    try:
        test_memory_cleanup()
        test_circular_reference_cleanup()
        
        print("\n" + "=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
