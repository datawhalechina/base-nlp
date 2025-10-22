"""Minimal, teaching-oriented Llama2 core implementation (single GPU/CPU, no MP).

Core modules under src/:
- rope: RoPE utilities (precompute/apply/repeat_kv)
- norm: RMSNorm
- attention: GroupedQueryAttention (KV cache + GQA)
- ffn: FeedForward (SwiGLU-style)
- transformer: TransformerBlock, LlamaTransformer
"""


