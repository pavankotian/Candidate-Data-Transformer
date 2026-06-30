# Multi-Source Candidate Data Transformer

A robust, fault-tolerant data engineering pipeline designed to ingest
candidate profiles from disparate structured (ATS JSON) and unstructured
(Recruiter Notes) sources, normalize field formats, resolve identity
clusters, and dynamically shape output schemas via runtime
configurations.

## Quick Start Guide

### 1. Installation

``` bash
pip install pydantic anthropic
```

### 2. Prepare Environment

Generate the mock input files and the runtime projection configuration:

``` bash
python generate_mock_inputs.py
python generate_runtime_config.py
```

### 3. Execution

Run the pipeline using the command-line interface:

``` bash
python main.py --ats ats_input.json --notes recruiter_notes.txt --config runtime_config.json
```

## System Highlights

-   **Deterministic Identity Resolution:** Uses primary key
    intersections (emails/phones) for reliable deduplication.
-   **Auditable Provenance:** Every field resolution decision is logged
    in an audit trail array.
-   **Dynamic Projection:** Remaps fields and applies safety protocols
    (null/omit/error) at runtime without modifying the pipeline core.
-   **Fault-Tolerant:** Isolated ingestion stages ensure that a single
    malformed source does not interrupt the entire processing batch.
