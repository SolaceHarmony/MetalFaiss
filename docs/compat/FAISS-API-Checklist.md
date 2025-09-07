# FAISS API Checklist (MetalFaiss Compat)

Feature | Status | Notes
---|---|---
IndexFlatL2/IP | Implemented | Wraps FlatIndex (L2/IP)
IndexIVFFlat | Implemented | L2 metric, nlist/nprobe
IndexIVFPQ | Implemented | ADC, L2 metric
IndexPQ | Implemented | ProductQuantizerIndex
IndexHNSWFlat | Implemented | efSearch/efConstruction exposed
IndexIDMap/IDMap2 | Implemented | ID mapping wrappers
IndexPreTransform | Implemented | Wraps PreTransformIndex
IndexRefineFlat | Implemented | Wraps RefineFlatIndex
IndexIVFOPQ | Stub | Compose OPQ→IVFPQ (planned)
IndexIVFScalarQuantizer | Stub/Shim | Implement SQ codebooks or route to Flat
IndexShards | Stub | Split add; device‑merge top‑k
IndexReplicas | Stub | Mirror add; merge top‑k
index_factory | Partial | Flat/FlatIP/IVF…,Flat/IVF…,PQ…/HNSW… (extend)
normalize_L2 | Implemented | Pure MLX
read_index/write_index | Partial | Use index_io; migrate to MLX‑only IO
range_search | Planned | Flat→IVF*/IVFPQ; pure MLX SearchRangeResult
remove_ids/reconstruct | Planned | Flat/IVF*/IVFPQ

Notes
- See PLAN.md for detailed tasks and upstream FAISS reference path.
- Wrappers return MLX arrays; avoid host conversions in hot paths.
