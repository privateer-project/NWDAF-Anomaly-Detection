# Dev Scripts - File Index

Quick navigation for the dev_scripts directory.

## Documentation Files

| File | Purpose | Lines |
|------|---------|-------|
| `README.md` | Main documentation with detailed descriptions | 95 |
| `QUICKREF.md` | Quick reference with troubleshooting tips | 150 |
| `SUMMARY.md` | Complete summary of what was created | 250 |
| `INDEX.md` | This file - navigation index | 50 |

## Test Scripts

| File | Tests | Lines | Runtime |
|------|-------|-------|---------|
| `test_utils.py` | Phase 1: Foundation utilities | 333 | ~2s |
| `test_serialization.py` | Phase 4: NumPy I/O | 338 | ~3s |
| `test_database.py` | Phase 3: SQLite & Repository | 294 | ~2s |
| `test_schema_registry.py` | Phase 5: Schema management | 255 | ~2s |
| `test_artifacts.py` | Phase 6: Model versioning | 405 | ~3s |
| `test_full_workflow.py` | Integration: Full pipeline | 351 | ~5s |

## Utilities

| File | Purpose |
|------|---------|
| `run_all.sh` | Bash runner for all tests with colored output |

## Usage Quick Start

```bash
# Run everything
./run_all.sh

# Run one test
python test_utils.py

# Get help
cat README.md
cat QUICKREF.md
```

## File Sizes

```
Documentation:  ~500 lines total
Test scripts:   ~2,000 lines total
Bash runner:    ~70 lines
Total:          ~2,600 lines
```

## Navigation

- **Need overview?** → Start with `README.md`
- **Need quick reference?** → See `QUICKREF.md`
- **Need details?** → See `SUMMARY.md`
- **Need to run tests?** → Use `run_all.sh` or individual scripts
- **Need this index?** → You're reading it!

---

Last Updated: 2025-11-04
