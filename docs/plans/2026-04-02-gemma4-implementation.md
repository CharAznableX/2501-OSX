# Gemma 4 Model Implementation Plan

## Model: Gemma-4-26B-A4B-it (MoE VL)
- Path: /Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M
- model_type: gemma4 / gemma4_text
- 26B total, 4B active (128 experts, top-8)
- 4-bit quantized, group_size 64

## Architecture Summary

### Text Model (30 layers)
- **Layer types**: `sliding_attention` (24) + `full_attention` (6)
  - Pattern: 5 sliding + 1 full, repeating 5 times
- **Attention**:
  - Sliding: 16 query heads, 8 KV heads, head_dim=256, sliding_window=1024
  - Full: 16 query heads, 2 global KV heads, global_head_dim=512, no window
  - QK normalization (q_norm, k_norm)
  - attention_k_eq_v: true (but separate weights exist)
- **RoPE**: Different per layer type
  - Sliding: theta=10000, default type
  - Full: theta=1000000, proportional type, partial_rotary_factor=0.25
- **FFN**: Dense MLP (GeGLU) + MoE switch_mlp per layer
  - Dense: intermediate_size=2112, gelu_pytorch_tanh
  - MoE: 128 experts, top-8, moe_intermediate_size=704
  - Router: proj + scale + per_expert_scale
- **Norms**: 6 RMSNorms per layer
  - input_layernorm, post_attention_layernorm
  - pre_feedforward_layernorm, post_feedforward_layernorm (dense MLP)
  - pre_feedforward_layernorm_2, post_feedforward_layernorm_1/2 (MoE)
- **Residual scaling**: `layer_scalar` per layer
- **Output**: final_logit_softcapping=30.0, tie_word_embeddings=true

### Weight Key Layout
```
model.language_model.embed_tokens.weight
model.language_model.norm.weight
model.language_model.layers.{i}.input_layernorm.weight
model.language_model.layers.{i}.layer_scalar
model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
model.language_model.layers.{i}.self_attn.{q,k}_norm.weight
model.language_model.layers.{i}.mlp.{gate,up,down}_proj.weight
model.language_model.layers.{i}.switch_mlp.{gate,up,down}_proj.weight  [128, ...]
model.language_model.layers.{i}.router.proj.weight
model.language_model.layers.{i}.router.scale
model.language_model.layers.{i}.router.per_expert_scale
model.language_model.layers.{i}.post_attention_layernorm.weight
model.language_model.layers.{i}.pre_feedforward_layernorm.weight
model.language_model.layers.{i}.post_feedforward_layernorm.weight
model.language_model.layers.{i}.pre_feedforward_layernorm_2.weight
model.language_model.layers.{i}.post_feedforward_layernorm_1.weight
model.language_model.layers.{i}.post_feedforward_layernorm_2.weight
```

### Vision Tower (for future VL support)
- 27 layers SigLIP-like
- patch_size=16, hidden_size=1152
- Separate implementation needed

## Implementation Steps

### Step 1: Configuration (Gemma4TextConfiguration)
- Decode from config.json text_config
- Fields: hiddenSize, numHiddenLayers, numAttentionHeads, numKeyValueHeads,
  numGlobalKeyValueHeads, headDim, globalHeadDim, intermediateSize, moeIntermediateSize,
  numExperts, topKExperts, vocabSize, rmsNormEps, slidingWindow, layerTypes,
  ropeParameters, finalLogitSoftcapping, tieWordEmbeddings, hiddenActivation

### Step 2: Gemma4Attention
- Separate implementations or mode switch for sliding vs full
- QK normalization before RoPE
- Different RoPE params per layer type
- Sliding window masking

### Step 3: Gemma4MoE (Router + SwitchMLP)
- Router: Linear(hidden_size, num_experts) with sigmoid + top-k selection
- Per-expert scale normalization
- SwitchMLP: [num_experts, hidden, intermediate] shaped expert weights
- Use existing SwitchLayers.swift patterns

### Step 4: Gemma4DecoderLayer
- input_layernorm -> attention -> post_attention_layernorm -> residual
- pre_feedforward_layernorm -> dense MLP -> post_feedforward_layernorm -> residual
- pre_feedforward_layernorm_2 -> MoE -> post_feedforward_layernorm_1/2 -> residual
- layer_scalar applied to residual

### Step 5: Gemma4TextModel
- embed_tokens -> layers -> norm
- logit_softcapping(30.0) on output
- tie_word_embeddings

### Step 6: Registry Integration
- Add "gemma4", "gemma4_text" to ModelRegistry
- Weight sanitization: strip "model.language_model." prefix
- newCache(): VMLXKVCacheSimple for all layers (no SSM)

### Step 7: Testing
- Load model, run inference
- Compare output quality with Python mlx-lm (once it adds gemma4)
- Benchmark speed

## Files to Create/Modify
- CREATE: `Packages/VMLXRuntime/Sources/VMLXRuntime/Models/Gemma4Model.swift` (~600 lines)
- MODIFY: `Packages/VMLXRuntime/Sources/VMLXRuntime/Models/ModelRegistry.swift` (add gemma4 case)
- MODIFY: `Packages/VMLXRuntime/Sources/VMLXRuntime/Models/WeightLoader.swift` (sanitize keys)
