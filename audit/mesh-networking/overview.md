# Mesh Networking Audit — Distributed Inference Across Apple Silicon

## Goal

Enable multiple Macs running the Osaurus app to collaborate on running models too large for a single machine. Similar to Exo (exo-explore/exo) and the concept of "LM Link". Each Mac contributes its Apple Silicon unified memory to collectively run a model that's been sharded by layers.

## Reference Projects

### Exo (Primary Reference)
- **Repo**: github.com/exo-explore/exo
- **Architecture**: Pipeline parallelism (layer sharding, NOT tensor parallelism)
- **Network**: mDNS/Bonjour discovery + gRPC (or raw TCP) for tensor transfer
- **Backend**: MLX-native on Apple Silicon
- **KV Cache**: Local per node (no cross-node sync needed)
- **Performance**: 80-90% of theoretical combined throughput on gigabit ethernet

### Why Pipeline Parallelism (Not Tensor Parallelism)
- Tensor parallelism splits individual matmul operations across nodes → requires NVLink-class microsecond latency
- Ethernet/WiFi has millisecond latency → tensor parallelism would 100x slow each operation
- Pipeline parallelism only transfers activation tensors at layer boundaries → one transfer per slab per token
- Activation tensor sizes are tiny: 8-32 KB per token (vs weight matrices at GB scale)

### Why Not Petals
- Petals is WAN-oriented (internet volunteers, unreliable nodes, DHT discovery)
- Overkill fault tolerance for a controlled LAN environment
- Not MLX-compatible (CUDA only)
- But: their dynamic load balancing patterns are instructive

---

## Architecture: 8 Layers

### Layer 1: Discovery (Bonjour/mDNS)

**Technology**: Apple's `Network.framework` — `NWBrowser` + `NWListener`

```
Service type: _osaurus._tcp
Port: dynamic (from NWListener)
```

**TXT Record Fields**:
```
device_id: UUID (stable per install)
device_name: "Eric's Mac Studio"
total_memory_gb: 192
available_memory_gb: 140
gpu_cores: 76 (M2 Ultra)
loaded_model: "Qwen3.5-72B-JANG-4bit" (or empty)
loaded_layers: "0-39" (or empty)
layer_capacity: 40 (how many layers this node can fit)
engine_version: "1.2.0"
status: "idle" | "loading" | "prefill" | "decode" | "sleeping"
cluster_id: UUID (matches coordinator's cluster)
```

**Swift Implementation**:
- `NWBrowser(for: .bonjour(type: "_osaurus._tcp", domain: nil))` — discovers peers
- `NWListener(using: .tcp)` — accepts connections from peers
- Automatic presence/absence detection when nodes join/leave network
- No manual IP configuration needed — pure zero-conf

**Functions Needed (New)**:
- `MeshDiscoveryManager` — actor managing NWBrowser + NWListener
  - `startBrowsing()` — begin discovering peers
  - `stopBrowsing()` — stop discovery
  - `advertiseSelf(capabilities:)` — publish own TXT record
  - `peers: [PeerNode]` — currently visible peers with capabilities
  - `onPeerDiscovered`, `onPeerLost` — callbacks

### Layer 2: Control Plane (Swift-to-Swift)

**Technology**: WebSocket via `NWConnection` (Network.framework) or URLSessionWebSocketTask

**Why WebSocket over gRPC**:
- Native Swift support, no external dependencies
- Bidirectional streaming for control messages AND tensor data
- Lower overhead than HTTP/2 for this use case
- Already using NWConnection for discovery → same framework

**Message Protocol** (binary header + payload):
```
[2 bytes: message_type] [4 bytes: payload_size] [4 bytes: sequence_num] [payload]
```

**Message Types**:
```
CLUSTER_JOIN = 0x01      // Peer wants to join cluster
CLUSTER_ACCEPT = 0x02    // Coordinator accepts peer
CLUSTER_LEAVE = 0x03     // Peer leaving cluster
SHARD_ASSIGN = 0x10      // Coordinator assigns layer range to peer
SHARD_LOADED = 0x11      // Peer confirms layers loaded
SHARD_UNLOAD = 0x12      // Coordinator tells peer to unload
INFERENCE_START = 0x20   // Coordinator starts inference pipeline
ACTIVATION_DATA = 0x21   // Activation tensor from previous stage
TOKEN_RESULT = 0x22      // Generated token(s) back to coordinator
CACHE_CLEAR = 0x30       // Clear KV cache on all nodes
CACHE_TRIM = 0x31        // Trim cache to N entries
HEALTH_PING = 0x40       // Heartbeat
HEALTH_PONG = 0x41       // Heartbeat response
STATUS_UPDATE = 0x42     // Node status change
SLEEP_COMMAND = 0x50     // Coordinator tells node to sleep
WAKE_COMMAND = 0x51      // Coordinator wakes node
```

**Functions Needed (New)**:
- `MeshControlPlane` — actor managing peer connections
  - `connect(to: PeerNode)` → NWConnection
  - `send(message:to:)` — send typed message to peer
  - `broadcastAll(message:)` — send to all peers
  - `onMessage(from:handler:)` — receive handler
  - `startHeartbeat(interval:)` — periodic health pings

### Layer 3: Cluster Coordination

**Role: Coordinator** (elected or designated)

**Election Strategy**: Simple — first node to start becomes coordinator. If coordinator dies, next oldest node takes over.

**Shard Assignment Algorithm**:
1. Coordinator reads model config: `num_hidden_layers`, parameter size per layer
2. Collects `available_memory_gb` from all peers
3. Allocates layers proportional to available memory:
   ```
   Node A (96GB available): layers 0-47  (60%)
   Node B (64GB available): layers 48-79 (40%)
   ```
4. Embedding layer + LM head go on coordinator (first and last stages)
5. Sends `SHARD_ASSIGN` to each peer with layer range

**Functions Needed (New)**:
- `MeshCoordinator` — actor (only active on coordinator node)
  - `formCluster(peers:model:)` → `ClusterPlan`
  - `assignShards(plan:)` → sends SHARD_ASSIGN to peers
  - `orchestrateInference(request:)` → manages pipeline flow
  - `handleNodeLoss(peer:)` → reassign or fail gracefully
  - `handleNodeJoin(peer:)` → potentially rebalance
- `ClusterPlan` — struct describing shard assignments
  - `nodes: [(peer: PeerNode, layerRange: Range<Int>)]`
  - `modelName: String`
  - `totalLayers: Int`

### Layer 4: Model Sharding & Loading

**vmlx-engine Modification Needed**: New CLI flag `--layer-range START-END`

```python
# In cli.py, add argument:
parser.add_argument("--layer-range", type=str, default=None,
                    help="Only load transformer layers in range START-END (e.g., 0-39)")

# In model loading:
if args.layer_range:
    start, end = map(int, args.layer_range.split("-"))
    # Filter safetensors to only load weights for layers[start:end]
    # Also load embedding (if start==0) and lm_head (if end==last)
```

**How MLX Model Loading Works**:
- Models use safetensors format with named tensors: `model.layers.0.self_attn.q_proj.weight`, etc.
- Filtering by layer index is straightforward: only `mx.load()` tensors matching the assigned range
- Embedding (`model.embed_tokens`) loads only on first node
- LM head (`lm_head`) loads only on last node

**Functions Needed (New Swift)**:
- `VMLXEngineConfig.buildArgs()` — add `--layer-range` flag when in mesh mode
- Model size estimator — estimate memory needed per layer range

**Functions Needed (New Python)**:
- `--layer-range` CLI argument parsing
- Selective layer loading in model initialization
- Activation input/output API for inter-node tensor passing

### Layer 5: Activation Tensor Transfer

**Data Flow (per token generation step)**:
```
[Coordinator: receives user request]
    ↓ (send prompt/token to Node A's engine)
[Node A: process layers 0-39, output activation]
    ↓ (16 KB float16 tensor over network)
[Node B: process layers 40-79, output logits]
    ↓ (sample token, send back)
[Coordinator: yield token to client SSE stream]
```

**Activation Tensor Sizes** (float16):
| Model | Hidden Dim | Per-Token (decode) | Per-1024-Token (prefill) |
|-------|-----------|-------------------|------------------------|
| 7B | 4096 | 8 KB | 8 MB |
| 13B | 5120 | 10 KB | 10 MB |
| 32B | 5120 | 10 KB | 10 MB |
| 70B | 8192 | 16 KB | 16 MB |
| 405B | 16384 | 32 KB | 32 MB |

**Network overhead** (gigabit ethernet, ~100 MB/s effective):
- 16 KB per token → 0.16ms latency per hop
- vs 10-50ms compute per layer slab
- **Network overhead: 1-3%** — negligible

**IPC Between Swift Gateway and Python Engine** (within same node):
- Option A: HTTP endpoint `/v1/pipeline/forward` — engine receives activation, processes layers, returns output activation
- Option B: Shared memory (mmap) — zero-copy tensor passing, fastest
- Option C: Unix domain socket — fast, no TCP overhead
- **Recommendation**: Start with HTTP endpoint (simplest), optimize to shared memory later

**Functions Needed (New)**:
- `MeshTensorTransfer` — handles activation tensor serialization/deserialization
  - `serialize(tensor: MLXArray) → Data` — raw bytes + shape/dtype header
  - `deserialize(data: Data) → (shape: [Int], dtype: DType, bytes: Data)`
- Engine endpoint: `POST /v1/pipeline/forward` — accepts activation tensor, returns processed tensor

### Layer 6: KV Cache in Mesh Mode

**Key Insight: KV cache stays LOCAL to each node.**

- Node A maintains KV cache for layers 0-39 only
- Node B maintains KV cache for layers 40-79 only
- No cross-node KV sync needed (this is the beauty of pipeline parallelism)

**Cache Coordination Needed**:
- `CACHE_CLEAR` broadcast: coordinator tells all nodes to clear cache (e.g., new conversation)
- `CACHE_TRIM` broadcast: coordinator tells all nodes to trim to same token count
- Session ID sync: all nodes use same session_id for consistent prefix cache
- **Functions**: Coordinator's `broadcastAll(CacheClear)`, nodes handle via existing `/v1/cache` DELETE endpoint

### Layer 7: Wake/Sleep Coordination

**Scenarios**:
1. **Node goes to sleep**: Bonjour advertisement disappears → coordinator detects loss → either:
   - Remaining nodes can fit full model → redistribute layers
   - Cannot fit → switch to smaller model or fail gracefully
   
2. **Node wakes up**: Bonjour re-advertises → coordinator detects → optionally rebalance:
   - If current model fits without new node → keep current assignment
   - If could run larger model with new node → offer to upgrade

3. **Coordinator tells node to sleep**: `SLEEP_COMMAND` → node soft-sleeps engine → frees GPU memory
4. **Coordinator wakes node**: `WAKE_COMMAND` → node wakes engine → reloads assigned layers

**Idle Management**:
- In single-node mode: existing idle timer → soft/deep sleep
- In mesh mode: coordinator manages cluster-wide idle timer
- All nodes sleep/wake together (can't run partial pipeline)
- Coordinator can send `POST /admin/deep-sleep` to all nodes simultaneously

**Functions Needed (New)**:
- `MeshCoordinator.handleNodeLoss(peer:)` — reassign or degrade
- `MeshCoordinator.handleNodeJoin(peer:)` — potentially rebalance
- `MeshCoordinator.sleepCluster()` / `wakeCluster()` — coordinated sleep/wake
- `MeshDiscoveryManager.onPeerLost` → triggers handleNodeLoss

### Layer 8: API Endpoint Routing

**Single API surface** — coordinator handles all external requests:
- `POST /v1/chat/completions` — received by coordinator's VMLXService
- Coordinator orchestrates the distributed pipeline internally
- SSE stream yielded back to client as if single-node
- **Clients are unaware of mesh topology**

**Gateway Enhancement**:
- Existing `OsaurusServer` on port 1337 stays as the entry point
- Existing `VMLXService.streamWithTools()` enhanced to handle mesh mode:
  - If model is mesh-sharded: route through `MeshCoordinator.orchestrateInference()`
  - If model is local-only: existing single-engine path (unchanged)

**Functions Modified**:
- `VMLXService.ensureEngineRunning()` — check if model needs mesh (too large for single node)
- `VMLXService.streamWithTools()` — branch: local engine vs mesh pipeline
- New: `MeshVMLXService` subclass or extension handling mesh-specific streaming

---

## Engine Modifications Needed (Python Side)

### New CLI Flags
```
--layer-range START-END        # Only load layers in range
--pipeline-mode                # Enable pipeline parallelism mode
--activation-input-port PORT   # Listen for incoming activations
--activation-output-host HOST  # Send output activations to next stage
--activation-output-port PORT  # Port for next stage
```

### New Endpoints
```
POST /v1/pipeline/forward      # Process activation tensor through loaded layers
GET  /v1/pipeline/info         # Return loaded layer range, memory usage
POST /v1/pipeline/reset        # Reset pipeline state (clear KV cache for assigned layers)
```

### Model Loading Changes
- `server.py` — load model with layer filter
- Safetensors selective loading: only weights matching assigned layer range
- Embedding/LM head loading based on position in pipeline

### Activation Format
```json
POST /v1/pipeline/forward
Content-Type: application/octet-stream
X-Shape: [1, 1, 8192]
X-DType: float16
X-Sequence-Id: "abc123"
X-Token-Position: 42

<raw bytes: 16384 bytes for [1,1,8192] float16>
```

Response: same format with processed activation.

---

## Swift Files Needed (New)

| File | Role |
|------|------|
| `Services/Mesh/MeshDiscoveryManager.swift` | Bonjour NWBrowser + NWListener, peer tracking |
| `Services/Mesh/MeshControlPlane.swift` | WebSocket connections, message protocol, heartbeat |
| `Services/Mesh/MeshCoordinator.swift` | Shard assignment, pipeline orchestration, cluster management |
| `Services/Mesh/MeshTensorTransfer.swift` | Activation tensor serialization, network transfer |
| `Services/Mesh/MeshVMLXService.swift` | VMLXService extension for mesh-mode inference |
| `Services/Mesh/PeerNode.swift` | Model: peer ID, capabilities, connection state |
| `Services/Mesh/ClusterPlan.swift` | Model: shard assignments, model config |
| `Services/Mesh/MeshMessage.swift` | Message types, encoding/decoding |
| `Views/Mesh/MeshStatusView.swift` | UI: cluster status, node list, shard visualization |
| `Views/Mesh/MeshSettingsView.swift` | UI: enable mesh, cluster name, auto-join toggle |

---

## Integration with Existing Systems

### With VMLXProcessManager
- In mesh mode, each node still spawns its own vmlx-engine subprocess
- ProcessManager needs to pass `--layer-range` flag
- Health checks still work per-node
- Restart logic: if one node's engine crashes, coordinator reassigns or waits for restart

### With VMLXGateway
- Gateway tracks local engine instances (unchanged)
- Mesh mode adds cluster-level tracking (which peer has which layers)

### With VMLXEngineConfig
- Add `--layer-range`, `--pipeline-mode` flags to `buildArgs()`
- Mesh-specific flags only added when coordinator assigns shard

### With InferenceProgressManager
- TPS computation changes: token emission rate at coordinator, not per-node
- TTFT includes network transfer time
- Could show per-node stats in mesh status view

### With Idle Sleep
- Coordinator manages cluster-wide idle timer
- `resetIdleTimer` on coordinator resets for all nodes
- Sleep/wake commands broadcast to all peers

### With JIT Compilation
- Each node handles JIT independently (MLX compiles kernels per-device)
- No cross-node JIT coordination needed
- First inference after shard load will be slower (JIT warmup)
- Coordinator should account for warmup time

### With Caching
- Prefix cache works per-node for assigned layers
- Session ID must be consistent across all nodes
- Cache clear/trim must be coordinated (broadcast)

---

## Performance Estimates

### Network Overhead per Token
| Network | Latency per hop | % of compute time | Viable? |
|---------|----------------|-------------------|---------|
| Thunderbolt 4 (40 Gbps) | 0.003ms | <0.1% | Excellent |
| 10GbE | 0.013ms | <0.1% | Excellent |
| Gigabit Ethernet | 0.16ms | 1-3% | Good |
| WiFi 6 (1.2 Gbps) | 0.32ms | 2-5% | Acceptable |
| WiFi 5 (866 Mbps) | 1-5ms | 5-15% | Marginal |

### Prefill Overhead (1024 tokens, 70B model, 2 nodes)
- Activation transfer: 16 MB over gigabit = 160ms
- Compute per node: ~2-5s
- Network overhead: ~3-8%

### Scaling Efficiency (Exo measurements)
- 2 nodes: ~90% efficiency
- 4 nodes: ~80% efficiency  
- 8 nodes: ~70% efficiency
- Diminishing returns beyond 4 nodes due to pipeline bubbles

---

## Development Phases

### Phase 1: Foundation (Discovery + Control Plane)
- Bonjour discovery (NWBrowser/NWListener)
- Peer connection management
- Heartbeat + status tracking
- Mesh status UI (node list)

### Phase 2: Sharding (Engine Modifications)
- `--layer-range` CLI flag in vmlx-engine
- Selective safetensors loading
- `/v1/pipeline/forward` endpoint
- Memory estimation per layer range

### Phase 3: Pipeline Orchestration
- Coordinator shard assignment
- Activation tensor transfer (HTTP initially)
- Token generation loop across nodes
- Integration with SSE streaming

### Phase 4: Resilience
- Node loss handling (graceful degradation)
- Node join handling (rebalancing)
- Coordinated sleep/wake
- Coordinator failover

### Phase 5: Optimization
- Shared memory IPC (replace HTTP for local node)
- Micro-batching for prefill
- Activation compression (optional)
- Speculative pipeline (overlap compute + transfer)

---

## Osaurus App Companion

The mesh networking feature aligns with having an "Osaurus app" on each Mac:
- Every Mac in the household/office runs Osaurus
- Bonjour discovers all instances automatically
- User picks a model → coordinator checks if single machine can handle it
- If not → offers to distribute across discovered peers
- Seamless UX: user just picks a model, mesh handles the rest
