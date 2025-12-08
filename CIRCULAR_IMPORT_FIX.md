# Circular Import Fix

**Date**: 2025-12-08
**Status**: ✅ **FIXED AND TESTED**
**Commit**: `9ed0b63`

---

## Problem

The fm-core-lib package had a circular import between models and clients:

```
fm_core_lib/__init__.py (line 15)
  → imports CaseServiceClient from clients
    → clients/case_service_client.py (line 6)
      → imports Case, Evidence from models
        → models/__init__.py
          → (circular back to fm_core_lib/__init__.py)
```

This caused `NameError: name 'Case' is not defined` when trying to import models.

---

## Root Cause Analysis

**The issue**: Both models AND clients were being imported at module initialization time in `__init__.py`:

```python
# Old code (BROKEN)
from fm_core_lib.models import Case, CaseStatus, ...
from fm_core_lib.clients import CaseServiceClient  # ← Triggers circular import
```

**Why it failed**:
1. When `__init__.py` runs, it tries to import `CaseServiceClient`
2. `CaseServiceClient` imports `Case` from models
3. But `Case` hasn't been fully defined yet because we're still in `__init__.py`
4. Result: `NameError`

---

## Solution: Lazy Import with `__getattr__`

Implemented lazy loading for `CaseServiceClient` to defer import until first access:

```python
# New code (FIXED)
# Export shared models first (no dependencies)
from fm_core_lib.models import (
    Case, CaseStatus, Evidence, Hypothesis, Solution,
    UploadedFile, InvestigationProgress, ConsultingData
)

# Export service discovery (no model dependencies)
from fm_core_lib.discovery import (
    ServiceRegistry,
    DeploymentMode,
    get_service_registry,
    reset_service_registry,
)

# Lazy import for clients to avoid circular dependency
# Clients depend on models, so import them last
def __getattr__(name):
    """Lazy import for CaseServiceClient to avoid circular import."""
    if name == "CaseServiceClient":
        from fm_core_lib.clients import CaseServiceClient
        return CaseServiceClient
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
```

**How it works**:
- Models are imported immediately (no circular dependency)
- `CaseServiceClient` is NOT imported at module initialization
- When code accesses `fm_core_lib.CaseServiceClient`, Python calls `__getattr__`
- `__getattr__` imports and returns `CaseServiceClient` on-demand
- By this time, models are already fully loaded, so no circular dependency

---

## Verification Tests

All tests passed:

### Test 1: Import models directly
```python
from fm_core_lib.models import Case
✓ SUCCESS
```

### Test 2: Import from top-level
```python
from fm_core_lib import Case, CaseStatus, Evidence
✓ SUCCESS
```

### Test 3: Lazy import client
```python
from fm_core_lib import CaseServiceClient
✓ SUCCESS (lazy loaded via __getattr__)
```

### Test 4: fm-case-service integration
```bash
cd /home/swhouse/product/fm-case-service/data
python3 test_case_creation.py
✅ All 5 tests PASSED
```

---

## Impact

### Before Fix:
```python
from fm_core_lib.models import Case  # ❌ NameError
```

### After Fix:
```python
from fm_core_lib.models import Case  # ✅ Works
from fm_core_lib import Case          # ✅ Works
from fm_core_lib import CaseServiceClient  # ✅ Works (lazy loaded)
```

---

## Related Files

- **Modified**: [`src/fm_core_lib/__init__.py`](src/fm_core_lib/__init__.py) - Added `__getattr__` for lazy imports
- **Tested with**: [`fm-case-service`](/home/swhouse/product/fm-case-service) - All integration tests pass

---

## References

- **Python PEP 562**: Module `__getattr__` and `__dir__`
- **Pattern**: Lazy imports to break circular dependencies
- **Commit**: `9ed0b63` - Fix circular import between models and clients

---

## Conclusion

The circular import issue is **completely resolved**. The fix:
- ✅ Breaks the circular dependency chain
- ✅ Preserves all public API functionality
- ✅ Requires no changes to consumer code
- ✅ Verified with integration tests

**All fm-core-lib imports now work without circular import errors.**
