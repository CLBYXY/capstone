# STEP 4 - Examination type mapping to LOINC

Target column: `type`

```
examination_events.csv
  |
  v
Phase 1 - text cleaning + time-noise filtering
    Some values of the type column are time placeholders such as "1 day".
    If a value contains a time expression (date_pattern)
    but no examination keyword (exam_keywords), it is treated as noise and set to None.
    Examples:
      "3 days ago"              -> removed
      "blood test 3 days ago"   -> kept
  |
  v
Phase 2 - building the LOINC reference space (SYSTEM cleaning + reference vocabulary)
    LOINC is a standard vocabulary of laboratory and clinical observations.
      column SYSTEM    : specimen or system, e.g. blood, urine
      column COMPONENT : what is measured, e.g. glucose tolerance
    Undesirable cases that need cleaning, e.g. SYSTEM "device", COMPONENT "machine 110".

    STEP 1  SYSTEM filter    - remove unsuitable SYSTEM values such as provider, device
    STEP 2  SYSTEM filter    - keep the top 500 of 2,588 SYSTEM classes (about 95% of records)
    STEP 3  SYSTEM normalise - 'bld' / 'blda' / 'bldv' -> 'blood'

    Some entries have a meaningless system but a meaningful component,
    e.g. system "patient", component "self-monitoring of blood glucose".

    STEP 4  build the reference vocabulary from component + system:
            if the system is meaningful, combine both; otherwise use the component only
  |
  v
Phase 3 - embedding (type vs component)
    SapBERT encodes the type column and the reference vocabulary from Phase 2
  |
  v
Phase 4 - FAISS similarity matching
  |
  v
examination_mapping.csv
```
