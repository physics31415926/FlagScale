# Data Preparation for FlagScale — Summary

## Overview

Prepare training data for FlagScale (Megatron backend). Two pipelines:
- **Pipeline A**: Text-only via `preprocess_data.py` → `.bin` + `.idx` files
- **Pipeline B**: Multimodal via Megatron-Energon → WebDataset `.tar` + TaskEncoder

## Key Decision: Which Pipeline?

| Use Case | Pipeline | Output Format |
|----------|----------|---------------|
| Text-only pretrain/SFT | A: preprocess_data.py | `.bin` + `.idx` |
| Multimodal (image/video + text) | B: Megatron-Energon | WebDataset `.tar` + Energon config |

## Data Pipeline Comprehension (MANDATORY)

Before writing any data processing code, trace the full pipeline:

```
Source Format → Processing Operations → Model Input Interface
```

| Link | What to Identify | Example Questions |
|------|-----------------|-------------------|
| **Source Format** | File format, schema, fields, modalities | JSONL with `text` field? WebDataset tar with `jpg` + `json`? |
| **Processing** | Tokenization, visual processing, label masking, packing | Which tokenizer? How are labels masked? Special tokens? |
| **Model Input** | `get_batch` signature, tensor shapes, dtypes | What keys does `forward()` expect? How are images interleaved? |

**MANDATORY**: Persist your pipeline understanding to memory before writing data code. An engineering gate enforces this.

## Pipeline A: Text-Only (bin + idx)

### Quick Start
1. Download demo data or prepare JSONL (one JSON per line with `text` field)
2. Run `preprocess_data.py` with tokenizer
3. Output: `<prefix>.bin` + `<prefix>.idx`
4. In training config: `data_path: ["/path/to/prefix"]` (no extension)

### Common Pitfalls
- Including `.bin` or `.idx` in `data_path` → fails
- Tokenizer mismatch between preprocessing and training → wrong vocab
- Missing `--append-eod` → documents concatenated without separator

## Pipeline B: Multimodal (Megatron-Energon)

### Quick Start
1. Organize data as WebDataset `.tar` files (image/video + text pairs)
2. Create Energon dataset config YAML
3. Write custom TaskEncoder (data loading logic)
4. In training config: `data_path: ["/path/to/energon_config.yaml"]`

### Key Concepts
- **WebDataset**: Tar archives with paired files (e.g., `00001.jpg` + `00001.json`)
- **TaskEncoder**: Python class that loads and processes each sample
- **Blending**: Mix multiple datasets with weights

### Common Pitfalls
- TaskEncoder doesn't match data format → KeyError or shape mismatch
- Missing `__restore_key__` in TaskEncoder → checkpoint resume fails
- Blend weights don't sum to 1.0 → unexpected sampling distribution

## Key Gates (Engineering Enforced)

- **Data Pipeline Gate**: Must trace source format → processing → model input and persist findings to memory before writing data code
- **Reading Depth**: Must read data-related source files before writing data processing code
- **Reading Quality**: Must cover source_format + processing + model_input categories

## Related Skills

- `train-config` — configure data paths in training YAML after data preparation
- `train-run` — launch training with prepared data
- `model-porter` — model porting (data pipeline understanding feeds into model input requirements)
