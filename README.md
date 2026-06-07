# Thesis Code Snippets

Representative code excerpts and sample data for the thesis:
*"Retrieval-Augmented Generation for Geospatial Question Answering over Danish Municipal Data"*

## Structure

```
thesis_code_snippets/
├── data/
│   ├── sample_collection/          Sample GeoJSON source file (3 features)
│   │   └── excavation_permits_sample.json
│   ├── sample_sentences/           Generated natural language sentences
│   │   └── excavation_permits_sentences_sample.json
│   └── ground_truth/               Evaluation questions
│       ├── ground_truth_sample.json   10-question representative sample
│       └── ground_truth_full.json     Full 199-question benchmark (global feature IDs)
│
├── templates/
│   └── sentence_templates_sample.json    Jinja2 templates for 3 collections
│
├── indexing/
│   ├── pageindex/
│   │   ├── schema_tree_sample.json              Property statistics for one dataset
│   │   └── collection_tree_enriched_excerpt.json Topic-grouped tree (one municipality)
│   ├── graph_indexed/
│   │   └── build_graph.py           Neo4j graph construction + enrichment
│   └── postgis/
│       └── ingest_postgis.py        GeoJSON → PostGIS table loader
│
├── retrieval/
│   ├── query_executor.py            Shared structured query engine (Python)
│   ├── vector_rag/
│   │   ├── search.py                Hybrid dense+sparse search with RRF fusion
│   │   └── ingest.py                Qdrant ingestion pipeline
│   ├── pageindex/                   (uses query_executor.py)
│   ├── graph_indexed/               (uses query_executor.py)
│   └── postgis/
│       └── spatial_query.py         PostGIS spatial SQL execution
│
└── evaluation/
    ├── judge_scoring.py             LLM-as-judge with structured tool_use output
    └── statistical_tests.py         Bootstrap CIs, paired tests, judge calibration
```

## Data

- **40 GeoJSON collections** from opendata.dk (5 Danish municipalities, ~303K features)
- **199 evaluation questions** across 5 categories: spatial_radius, spatial_nearest, property_filter, conversational, unanswerable
- The complete benchmark is provided in `data/ground_truth/ground_truth_full.json`: all 199 questions, each with its category, expected answer, source collection, and verified global feature IDs. `ground_truth_sample.json` is a 10-question subset for a quick look.
- Sample collection and sentence files contain representative excerpts, not the full datasets

## Requirements

- Python 3.11+
- PostgreSQL 18 with PostGIS 3.6
- Neo4j (community edition)
- Qdrant (local embedded mode)
- bge-m3 embedding model (BAAI/bge-m3)
