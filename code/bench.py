import torch, time, os, sys
from transformers import AutoModel
os.environ.setdefault('HF_HUB_OFFLINE', '1')
names = sys.argv[1:] or ['BAAI/bge-reranker-v2-m3', 'microsoft/deberta-v3-large', 'microsoft/deberta-v3-base']
for n in names:
    m = AutoModel.from_pretrained(n).cuda()
    m.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
    for L, bs in [(256, 20), (192, 20)]:
        x = torch.randint(5, 1000, (bs, L)).cuda()
        att = torch.ones_like(x)
        for it in range(6):
            if it == 2:
                torch.cuda.synchronize(); t = time.time()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = m(input_ids=x, attention_mask=att).last_hidden_state
            h.float().mean().backward(); opt.step(); opt.zero_grad()
        torch.cuda.synchronize()
        print(n, L, bs, 'train pairs/s', round(4 * bs / (time.time() - t), 1), 'mem', torch.cuda.max_memory_allocated() // 2**20, flush=True)
    del m, opt
    torch.cuda.empty_cache()
