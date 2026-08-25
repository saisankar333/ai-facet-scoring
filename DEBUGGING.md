# DEBUGGING.md — AI Facet Scoring

This document records the actual debugging issues encountered while developing and testing the AI Facet Scoring pipeline.

## 1. Qwen Response Contained Separate Reasoning Content

### Problem

During testing with `Qwen/Qwen3-8B`, the Hugging Face response contained separate final-answer and reasoning fields.

The final structured JSON was available in `message.content`, while `message.reasoning_content` contained model reasoning.

### Why This Mattered

The pipeline requires structured JSON containing:

- `status`
- `evidence_strength`
- `evidence`
- `reason`

Model reasoning must not be treated as the final answer or as evidence.

### Fix

The scoring implementation reads only the final `content` field:

```python
content = getattr(
    message,
    "content",
    None,
)
