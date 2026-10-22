# STEP 5 - Further-examination need mapping to LOINC

Target column: `test_need`

The process is close to STEP 4 (examination_events.csv). The main differences:

1. Phase 2 uses a stricter SYSTEM filter: values such as "patient" are excluded and only
   test-related systems are kept.
2. A different matching rule:
   - common systems are grouped into lists, e.g. "blood";
   - if the need text contains a word from such a list, only LOINC records with that
     SYSTEM are searched;
   - otherwise a global similarity search is used.

```
examination_need.csv (test_need)
  -> Phase 1: text normalisation (lower-case)
  -> Phase 2: LOINC reference space (SYSTEM cleaning + reference vocabulary)
  -> Phase 3: embedding
  -> Phase 4: FAISS similarity matching
  -> need_mapping.csv
```
