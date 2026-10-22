# STEP 3 - Symptom duration cleaning

Target column: `time`

```
symptom_events.csv
  |
  v
Phase 1 - rule-based cleaning (semantic normalisation + noise filtering)
    CURRENT + RECENT                                        -> recent
    TODAY + YESTERDAY                                       -> 1 day
    NAN / NOT SPECIFIED / NOT STATED / UNSPECIFIED / NOT PROVIDED -> none
  |
  v
Phase 2 - structuring time expressions (digits + extraction + unit unification)
    rows handled in Phase 1 are excluded
    two -> 2
    extract the pattern "<number> <time word>", e.g. "3 year"
    unify units, e.g. 90 day -> 3 month
  |
  v
Phase 3 - semantic labels (occasional / ongoing / long-standing)
    rows handled in Phases 1-2 are excluded
    keyword "occasional" -> "occasional"
    keyword "ongoing"    -> "ongoing"
  |
  v
Phase 4 - removal of uninformative words (since / month names)
    rows handled in Phases 1-3 are excluded
    expressions such as "since 2020" or "Jan" carry no duration and are set to none
  |
  v
Phase 5 - semantic clustering (embedding + DBSCAN)
    remaining expressions, e.g. "a few days", "short period", "some time"
    encoder: all-MiniLM-L6-v2
    clustering: DBSCAN (cluster_name = expression closest to the cluster centroid)
    cluster != -1 -> replaced by cluster_name
    cluster == -1 -> NaN
  |
  v
symptom_cleaned.csv
```
