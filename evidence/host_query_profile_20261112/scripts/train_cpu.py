"""CPU replica of tools.large_ngram_application.train (exact addressing only)."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from torch import nn
sys.path.insert(0, '/Users/duyimin/paper_work/ngram-store')
from tools.large_ngram_application import batch_csr

data, out = Path(sys.argv[1]), Path(sys.argv[2]); lr0 = float(sys.argv[3]); seed = 3; INIT = sys.argv[4] if len(sys.argv) > 4 else 'zero'
epochs, batch, dim = 10, 512, 128
torch.set_num_threads(8); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
registry = np.load(data / 'registry.npy', mmap_mode='r'); rows = len(registry) + 1
ids = np.load(data / 'train-ids.npy', mmap_mode='r'); ptr = np.load(data / 'train-offsets.npy', mmap_mode='r')
labels = np.load(data / 'train-labels.npy'); classes = int(labels.max()) + 1
emb = nn.EmbeddingBag(rows, dim, mode='mean', sparse=True, include_last_offset=True, padding_idx=0)
clf = nn.Linear(dim, classes)
with torch.no_grad():
    (emb.weight.zero_() if INIT=='zero' else (nn.init.uniform_(emb.weight,-1/dim,1/dim,generator=torch.Generator().manual_seed(seed+100003)), emb.weight[0].zero_()))
    g = torch.Generator().manual_seed(seed); nn.init.normal_(clf.weight, std=.1, generator=g); clf.bias.zero_()
opt = torch.optim.SGD([*emb.parameters(), *clf.parameters()], lr=lr0)
rng = np.random.default_rng(seed); steps = (len(labels) + batch - 1) // batch; t0 = time.perf_counter()
for epoch in range(epochs):
    perm = rng.permutation(len(labels)); losses = []
    for step, b in enumerate(range(0, len(labels), batch)):
        docs = perm[b:b + batch]; flat, off = batch_csr(ids, ptr, docs)
        opt.param_groups[0]['lr'] = lr0 * (1 - (epoch * steps + step) / (epochs * steps))
        loss = nn.functional.cross_entropy(clf(emb(torch.from_numpy(flat), torch.from_numpy(off))), torch.from_numpy(labels[docs]))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss))
    print(json.dumps({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'seconds': round(time.perf_counter() - t0, 1)}), flush=True)
out.mkdir(parents=True, exist_ok=True)
np.save(out / 'rows.npy', emb.weight.detach().numpy())
np.savez(out / 'classifier.npz', weight=clf.weight.detach().numpy(), bias=clf.bias.detach().numpy())
print('done', rows, flush=True)
