# STEP 1 - Disease normalisation

```
Raw disease text
  -> [cleaning] lower-case
  -> SapBERT embedding (biomedical encoder)
  -> [UMLS] filter semantic types with MRSTY, keep T047 / T048 / T191
  -> disease CUI
  -> [UMLS] look up the preferred name of the CUI in MRCONSO
  -> [cleaning] fix inverted terms
  -> standard disease name (fine-grained)
  -> SapBERT embedding
  -> KMeans clustering
  -> standard disease name (coarse-grained, 25 clusters)
```
