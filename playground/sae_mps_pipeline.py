# small_instruct_sae_mps_pipeline.py (with tqdm)
import os, math, heapq, threading, queue
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# =========================
# Config (pick your model)
# =========================
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
DATASET = os.environ.get("DATASET", "chrisociepa/wikipedia-pl-20230401")
STREAM_SPLIT = os.environ.get("STREAM_SPLIT", "train")

MAX_LEN   = int(os.environ.get("MAX_LEN",   "2048"))
STRIDE    = int(os.environ.get("STRIDE",    "1024"))
PREFETCH  = int(os.environ.get("PREFETCH",  "8"))

MICRO_TOKENS = int(os.environ.get("MICRO_TOKENS", "160"))
ACCUM        = int(os.environ.get("ACCUM",        "8"))
LR           = float(os.environ.get("LR",         "3e-3"))
TRAIN_STEPS  = int(os.environ.get("TRAIN_STEPS",  "50"))
PRINT_EVERY  = int(os.environ.get("PRINT_EVERY",  "500"))

SCORE_BLOCKS   = int(os.environ.get("SCORE_BLOCKS",   "5"))
COLLECT_BLOCKS = int(os.environ.get("COLLECT_BLOCKS", "5"))
TOP_FEATURES   = int(os.environ.get("TOP_FEATURES",   "200"))
PER_FEAT_EX    = int(os.environ.get("PER_FEAT_EX",    "40"))
CONTEXT_TOK    = int(os.environ.get("CONTEXT_TOK",    "28"))
TAU_Q          = float(os.environ.get("TAU_Q",        "0.95"))

# =========================
# MPS setup
# =========================
assert torch.backends.mps.is_available(), "MPS not available."
device = torch.device("mps")
DTYPE  = torch.float16
torch.set_float32_matmul_precision("medium")
try:
    torch.mps.empty_cache()
except Exception:
    pass

# =========================
# Model + tokenizer
# =========================
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
).to(device).eval()
for p in model.parameters(): p.requires_grad_(False)

def get_num_layers(m):
    try: return len(m.lm.layers)
    except Exception: return getattr(m.config, "num_hidden_layers", 24)

N_LAYERS = get_num_layers(model)
TARGET_LAYER = int(os.environ.get("TARGET_LAYER", str(max(0, N_LAYERS // 2))))
print(f"[info] Using {MODEL_ID} with {N_LAYERS} layers; TARGET_LAYER={TARGET_LAYER}")

# =========================
# Streaming + prefetch
# =========================
def block_stream():
    ds = load_dataset(DATASET, split=STREAM_SPLIT, streaming=True)
    for ex in ds:
        ids = tok(ex["text"], return_tensors="pt", truncation=False, add_special_tokens=False).input_ids[0]
        for s in range(0, len(ids), STRIDE):
            chunk = ids[s:s+MAX_LEN]
            if chunk.numel() > 8:
                yield chunk.unsqueeze(0)

def prefetch_blocks(maxq=PREFETCH):
    q = queue.Queue(maxsize=maxq)
    stop = object()
    def worker():
        for b in block_stream():
            q.put(b)
        q.put(stop)
    t = threading.Thread(target=worker, daemon=True); t.start()
    while True:
        item = q.get()
        if item is stop: break
        yield item

# =========================
# Hook
# =========================
_act = None
def _grab_hook(_m, _i, out):
    global _act
    _act = out.detach()

hook_handle = model.model.layers[TARGET_LAYER].register_forward_hook(_grab_hook)

# =========================
# SAE
# =========================
class TopKSAE(nn.Module):
    def __init__(self, d_in, d_lat, k):
        super().__init__()
        self.enc = nn.Linear(d_in, d_lat, bias=True)
        self.dec = nn.Linear(d_lat, d_in, bias=True)
        self.dec.weight = nn.Parameter(self.enc.weight.T)
        self.k = k

    def forward(self, x):
        pre = self.enc(x)
        vals, idx = torch.topk(pre, k=self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(dim=-1, index=idx, src=torch.relu(vals))
        return self.dec(z), z

def pick_sae_sizes(d_model):
    name = MODEL_ID.lower()
    if "0.5b" in name: return 4*d_model, 48
    if "1.1b" in name or "1.5b" in name: return 4*d_model, 64
    if "phi-3" in name: return 5*d_model, 64
    return 4*d_model, 64

sae, opt = None, None

# =========================
# Training
# =========================
def train_sae():
    global sae, opt
    step = 0
    with tqdm(total=TRAIN_STEPS, desc="Training SAE", unit="step") as pbar:
        for block in prefetch_blocks():
            if step >= TRAIN_STEPS: break
            with torch.no_grad():
                _ = model(input_ids=block.to(device))
            H = _act.squeeze(0)
            T, D = H.shape
            if sae is None:
                SAE_LATENTS, TOPK = pick_sae_sizes(D)
                print(f"[info] d_model={D}, SAE_LATENTS={SAE_LATENTS}, TOPK={TOPK}")
                sae = TopKSAE(D, SAE_LATENTS, TOPK).to(device, dtype=DTYPE)
                opt = optim.AdamW(sae.parameters(), lr=LR)

            perm = torch.randperm(T, device=device)
            running = 0
            for i in range(0, T, MICRO_TOKENS):
                idx = perm[i:i+MICRO_TOKENS]
                xb = H.index_select(0, idx)
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    xhat, z = sae(xb)
                    recon = ((xhat - xb)**2).mean()
                (recon / ACCUM).backward()
                running += 1
                if running % ACCUM == 0:
                    opt.step(); opt.zero_grad(set_to_none=True)
                    step += 1; pbar.update(1)
                    if step % PRINT_EVERY == 0:
                        pbar.set_postfix({"recon": f"{float(recon):.6f}"})
                    if step >= TRAIN_STEPS: break
    return sae

# =========================
# Feature scoring
# =========================
def score_features(n_blocks=SCORE_BLOCKS, tau_q=TAU_Q):
    sums, counts, maxes, fires = defaultdict(float), defaultdict(int), defaultdict(float), defaultdict(int)
    total_tokens = 0
    with torch.no_grad():
        for _ in tqdm(range(n_blocks), desc="Scoring feats", unit="blk"):
            block = next(prefetch_blocks())
            _ = model(input_ids=block.to(device))
            H = _act.squeeze(0); T,_D=H.shape
            _, Z = sae(H); Z32=Z.to(torch.float32)
            total_tokens += T
            tau = torch.quantile(Z32.flatten(), tau_q).item()
            s = Z32.sum(0); m = Z32.max(0).values; f=(Z32>tau).sum(0)
            for j in range(s.shape[0]):
                sums[j]+=float(s[j]); counts[j]+=T; fires[j]+=int(f[j])
                if float(m[j])>maxes[j]: maxes[j]=float(m[j])
    return {k:{"mean":sums[k]/max(1,counts[k]),"max":maxes[k],"fire_rate":fires[k]/max(1,total_tokens)} for k in sums}

# =========================
# Collect examples
# =========================
def collect_examples(top_features, per_feat=PER_FEAT_EX, context_tokens=CONTEXT_TOK,
                     k_local=8, n_blocks=COLLECT_BLOCKS, tau_q=TAU_Q):
    heaps={k:[] for k in top_features}; tops={k:Counter() for k in top_features}
    def push(k,score,span):
        h=heaps[k]; item=(float(score),span.cpu().clone())
        if len(h)<per_feat: heapq.heappush(h,item)
        elif score>h[0][0]: heapq.heapreplace(h,item)
    with torch.no_grad():
        for _ in tqdm(range(n_blocks), desc="Collecting ex", unit="blk"):
            block = next(prefetch_blocks())
            ids=block[0]; _=model(input_ids=block.to(device))
            H=_act.squeeze(0); T,_=H.shape
            _,Z=sae(H); Z32=Z.to(torch.float32)
            tau=torch.quantile(Z32.flatten(),tau_q).item()
            vals,idxs=torch.topk(Z32,k=min(k_local,Z32.shape[1]),dim=-1)
            for t in range(T):
                a=max(0,t-context_tokens); b=min(T,t+1+context_tokens)
                span=ids[a:b]
                for j in range(idxs.shape[1]):
                    kf=int(idxs[t,j]); s=float(vals[t,j])
                    if kf not in heaps or s<=tau: continue
                    push(kf,s,span); tops[kf][int(ids[t])]+=1
    return heaps,tops

# =========================
# Main
# =========================
if __name__=="__main__":
    print("[stage] Training SAE…")
    train_sae(); torch.mps.empty_cache()

    print("[stage] Scoring features…")
    stats=score_features()
    ranked=sorted(stats.keys(),key=lambda k:stats[k]["max"]*math.sqrt(max(1e-9,stats[k]["fire_rate"])),reverse=True)
    focus=ranked[:TOP_FEATURES]

    print("[stage] Collecting examples…")
    heaps,tops=collect_examples(focus)

    print("\n[report] Top 5 features")
    for k in focus[:5]:
        card={"id":k,"mean":stats[k]["mean"],"max":stats[k]["max"],"fire_rate":stats[k]["fire_rate"]}
        print(f"\n=== Feature {card['id']} === mean={card['mean']:.4f} max={card['max']:.4f} fire={100*card['fire_rate']:.2f}%")
        top_tokens=[(tok.decode([tid],skip_special_tokens=True) or tok.convert_ids_to_tokens(tid),c) for tid,c in tops[k].most_common(10)]
        print("Tokens:",top_tokens)
        snippets=sorted(heaps[k],key=lambda x:x[0],reverse=True)[:5]
        for _,span in snippets:
            print("  …"+tok.decode(span,skip_special_tokens=True).replace("\n"," ")[:160]+"…")