"""Reviewer checks on the grid-map / collapse claims.

(A) Neural-collapse test: effective rank of the last hidden layer + TRAIN accuracy + isolate
    fraction, per dataset, both arms. Hypothesis: the ungated net collapses (low eff-rank, high
    isolates) only where training fits easily (few classes), not on 47-class EMNIST.
(B) Is the gated ring architectural or learned? active-map coherence/alignment for the gated net
    at RANDOM INIT vs trained (MNIST).
(C) How much shape do the gate windows explain alone? 'data-only' map = variance across images of
    the B5/S3 response to the raw image, no network (MNIST). Ring or blob?

Outputs printed + diagnostics/er_reviewer_maps.png (untrained | trained | data-only, MNIST).
Run: PYTHONPATH=. python paper_code/er_reviewer_checks.py
"""
from conwaynet import REPO_ROOT
import json, numpy as np, torch, networkx as nx
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model
from conwaynet.lateral_inhibition import moore_mean, _rule_mask
ROOT=REPO_ROOT; DEV="cuda" if torch.cuda.is_available() else "cpu"
UNITS=100; TD=6.0; G=28; N=1000
DS=[("MNIST","mnist"),("KMNIST","kmnist"),("FashionMNIST","fashionmnist"),("EMNIST","emnist")]

def acts(m,Xe):
    x=Xe.view(Xe.size(0),-1); out=[]
    with torch.no_grad():
        for i,lin in enumerate(m.linears):
            t=lin(x); t=m.norms[i](t) if m.norms is not None else t; z=torch.tanh(t)
            if m.gate is not None: z=m.gate(x,z)
            x=z; out.append(z.detach().cpu().numpy())
    return out
def eff_rank(a):  # participation ratio of the activation covariance
    C=np.cov(a.T); w=np.linalg.eigvalsh(C); w=w[w>0]; return float((w.sum()**2)/(w**2).sum())
def active_map(A):
    cols,uid=[],[]
    for a in A:
        idx=np.argsort(a.var(0))[-min(UNITS,a.shape[1]):]; cols.append(a[:,idx]); uid+=list(idx)
    X=np.concatenate(cols,1); C=np.corrcoef(X.T); np.fill_diagonal(C,0.); C=np.nan_to_num(C)
    n=C.shape[0]; iu=np.triu_indices(n,1); mg=np.abs(C[iu]); thr=np.sort(mg)[-int(TD*n/2)]
    Gr=nx.Graph(); Gr.add_nodes_from(range(n)); Gr.add_edges_from([(int(a),int(b)) for a,b,s in zip(iu[0],iu[1],mg>=thr) if s])
    mp=np.zeros((G,G)); iso=0
    for v in Gr:
        if Gr.degree(v)>0: r,c=divmod(int(uid[v]),G); mp[r,c]+=1
        else: iso+=1
    return mp, iso/Gr.number_of_nodes()
def coh(mp):
    nb=(np.roll(mp,1,0)+np.roll(mp,-1,0)+np.roll(mp,1,1)+np.roll(mp,-1,1))/4
    a=mp.ravel()-mp.mean(); b=nb.ravel()-nb.mean(); return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-12))
def align(mp,dv):
    a=mp.ravel()-mp.mean(); b=dv.ravel()-dv.mean(); return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-12))
def train_acc(m,tr):            # over the full training split (eval mode)
    m.eval(); cor=tot=0
    with torch.no_grad():
        for xb,yb in tr:
            p=m(xb.to(DEV)); p=p[0] if isinstance(p,tuple) else p
            cor+=(p.argmax(1).cpu()==yb).sum().item(); tot+=len(yb)
    return cor/tot

# ---------- (A) per-dataset collapse table ----------
print("(A) NEURAL-COLLAPSE TEST (last-hidden effective rank, train acc, isolates):")
print(f"{'dataset':12s} {'#cls':>4s} | {'effrank_ca':>10s} {'effrank_none':>12s} | {'trainacc_ca':>11s} {'trainacc_none':>13s} | {'iso_ca':>6s} {'iso_none':>8s}")
for dn,ds in DS:
    c_ca=load_config(f"{ROOT}/configs/paper/tanh/fcn_{ds}_tanh_ns_ki_ca.json")
    tr,va,te,meta=load_dataset(c_ca["data_path"],c_ca["batch_size"],c_ca["data_name"],val_frac=c_ca["val_frac"],
        split_seed=c_ca["split_seed"],data_norms=c_ca["data_norms"],output_path=None,num_workers=2)
    Xe=next(iter(te))[0][:N].to(DEV)
    row={}
    for mech in ("ca","none"):
        c=load_config(f"{ROOT}/configs/paper/tanh/fcn_{ds}_tanh_ns_ki_{mech}.json")
        m=build_model(c,meta).to(DEV); m.load_state_dict(torch.load(f"{ROOT}/paper_runs/fcn_{ds}_tanh_ns_ki_d10/model_{mech}.pth",map_location=DEV)); m.eval()
        A=acts(m,Xe); er=eff_rank(A[-1]); _,iso=active_map(A); ta=train_acc(m,tr)
        row[mech]=(er,iso,ta)
    print(f"{dn:12s} {meta['num_classes']:4d} | {row['ca'][0]:10.1f} {row['none'][0]:12.1f} | {row['ca'][2]:11.3f} {row['none'][2]:13.3f} | {row['ca'][1]:6.2f} {row['none'][1]:8.2f}",flush=True)

# ---------- (B) untrained vs trained gated map (MNIST) + (C) data-only map ----------
c_ca=load_config(f"{ROOT}/configs/paper/tanh/fcn_mnist_tanh_ns_ki_ca.json")
tr,va,te,meta=load_dataset(c_ca["data_path"],512,c_ca["data_name"],val_frac=c_ca["val_frac"],
    split_seed=c_ca["split_seed"],data_norms=c_ca["data_norms"],output_path=None,num_workers=0)
Xe=next(iter(te))[0][:N].to(DEV); dv=Xe.view(-1,G,G).var(0).cpu().numpy()
torch.manual_seed(0); np.random.seed(0)
m_init=build_model(c_ca,meta).to(DEV); m_init.eval()                      # random init, untrained
mp_init,iso_init=active_map(acts(m_init,Xe))
m_tr=build_model(c_ca,meta).to(DEV); m_tr.load_state_dict(torch.load(f"{ROOT}/paper_runs/fcn_mnist_tanh_ns_ki_d10/model_ca.pth",map_location=DEV)); m_tr.eval()
mp_tr,iso_tr=active_map(acts(m_tr,Xe))
# (C) data-only: variance across images of the B5/S3 response to the raw image (no network)
xo=((Xe.view(-1,1,G,G)+1.0)/2.0); mm=moore_mean(xo)
resp=(_rule_mask(mm,[5.0],c_ca["delta"],c_ca["gamma"])+_rule_mask(mm,[3.0],c_ca["delta"],c_ca["gamma"]))
mp_data=resp.var(0).squeeze().cpu().numpy()
print("\n(B) GATED MAP architectural vs learned (MNIST):")
print(f"    untrained: coherence={coh(mp_init):.2f} align={align(mp_init,dv):.2f} iso={iso_init:.2f}")
print(f"    trained  : coherence={coh(mp_tr):.2f} align={align(mp_tr,dv):.2f} iso={iso_tr:.2f}")
print(f"(C) DATA-ONLY gate response: coherence={coh(mp_data):.2f} align={align(mp_data,dv):.2f}")
fig,axs=plt.subplots(1,4,figsize=(14,3.6))
for ax,mp,t in [(axs[0],mp_init,f"gated UNTRAINED\niso {iso_init:.2f}"),(axs[1],mp_tr,f"gated TRAINED\niso {iso_tr:.2f}"),
                (axs[2],mp_data,"gate on DATA only\n(no network)"),(axs[3],dv,"data pixel-variance")]:
    ax.imshow(mp,cmap="magma"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t,fontsize=11)
fig.tight_layout(); fig.savefig(f"{ROOT}/diagnostics/er_reviewer_maps.png",dpi=140)
print("WROTE diagnostics/er_reviewer_maps.png")
print("REVIEWER_CHECKS_DONE")
