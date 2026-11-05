"""Occupancy distribution per clean paper net (CNN = no-stem CNN with the per-channel image gate; at that
gate the occupancy recorded is the one the rule reads: each channel |m - median| / max): pooled mean / std / sparsity-fraction of the field
m = Moore-mean((x+1)/2) over all gate applications × spatial positions × samples. Tests whether the
low-occupancy regime is uniform suppression or sparse (most of the field near 0). Revised barplot
with std error bars + sparsity annotation. On the grayscale MNIST-family nets the field is also split into object
and background (object = gate-grid positions whose 3x3 image neighbourhood has ink > 0.1, the paper's dilated mask,
centre-cropped to each gate's grid), drawn as two thin bars."""
from conwaynet import REPO_ROOT
import json, numpy as np, torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from conwaynet.config import load_config
from conwaynet.models import build_model
from conwaynet.lateral_inhibition import LateralInhibitionCA, GraphLateralInhibitionCA
from conwaynet.data import load_dataset
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = REPO_ROOT; MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json")); SPARSE_THR = 0.10
NETS = [("MLP","mnist","fcn_mnist_tanh_ns_ki_d10"),("MLP","fashion","fcn_fashionmnist_tanh_ns_ki_d10"),
        ("MLP","kmnist","fcn_kmnist_tanh_ns_ki_d10"),("MLP","emnist","fcn_emnist_tanh_ns_ki_d10"),
        ("CNN","mnist","cnn_mnist_perch"),("CNN","fashion","cnn_fashionmnist_perch"),
        ("CNN","kmnist","cnn_kmnist_perch"),("CNN","emnist","cnn_emnist_perch"),
        ("CNN","mcolor","cnn_mnist_color_perch"),("CNN","fcolor","cnn_fashionmnist_color_perch"),
        ("CNN","kcolor","cnn_kmnist_color_perch"),("CNN","ecolor","cnn_emnist_color_perch"),
        ("WRN","cifar10","wrn_cifar10"),("GCN","mnist","gcn2_mnist/s0")]
ACOL = {"MLP":"#d1495b","CNN":"#30638e","WRN":"#3a7d44","GCN":"#8e5ea2"}
GRAY = {"mnist","fashion","kmnist","emnist"}          # black-background datasets: object / background split
# config of each net: the paper map, plus the GCN (paper_runs/gcn2_mnist, seed 0)
CFG = {**{k: v[0] for k, v in MAP.items()}, "gcn2_mnist/s0": f"{ROOT}/configs/gcn2_mnist/gcn2_mnist_s0_ca.json"}

def moore_mean(x):
    s=(torch.roll(x,1,-1)+torch.roll(x,-1,-1)+torch.roll(x,1,-2)+torch.roll(x,-1,-2)
       +torch.roll(x,(1,1),(-2,-1))+torch.roll(x,(1,-1),(-2,-1))+torch.roll(x,(-1,1),(-2,-1))+torch.roll(x,(-1,-1),(-2,-1)))
    return s/8.0

@torch.no_grad()
def occ_stats(run, split):
    cfg=load_config(CFG[run]); rd=f"{ROOT}/paper_runs/{run}"
    tr,va,te,meta=load_dataset(cfg["data_path"],cfg["batch_size"],cfg["data_name"],val_frac=cfg["val_frac"],
        split_seed=cfg["split_seed"],data_norms=cfg["data_norms"],output_path=rd,num_workers=2)
    m=build_model(cfg,meta).to(DEV); m.load_state_dict(torch.load(f"{rd}/model_ca.pth",map_location=DEV)); m.eval()
    xb,_=next(iter(te)); X=xb[:250].to(DEV)
    acc={r:{"n":0,"s":0.0,"s2":0.0,"sp":0} for r in ("all","obj","bg")}
    mean_,std_=(torch.tensor(v,dtype=torch.float32,device=DEV).view(1,-1,1,1) for v in cfg["data_norms"])
    ink=F.max_pool2d(((X*std_+mean_).mean(1,keepdim=True)>0.1).float(),3,1,1)   # dilated object mask on the image grid
    hooks=[]
    for name,mod in m.named_modules():
        if isinstance(mod,(LateralInhibitionCA,GraphLateralInhibitionCA)):
            def mk(md):
                def h(mo,inp,o):
                    x=inp[0]
                    if x.dim()==2: x=x.view(x.shape[0],md.channels,md.height,md.width)
                    # the GCN gate's m is the mean over the node's graph neighbours (no wraparound), as it reads it
                    mm=md.neighbour_mean((x+1.0)/2.0) if isinstance(md,GraphLateralInhibitionCA) else moore_mean((x+1.0)/2.0)
                    if md.cfg["channel_reduce"]=="perch_bgref" and x.shape[1]!=o.shape[1]:
                        # per-channel image gate: the rule reads each channel's background-referenced occupancy
                        f=mm.flatten(2); med=f.median(-1,keepdim=True).values; dv=(f-med).abs()
                        mm=(dv/dv.amax(-1,keepdim=True).clamp_min(1e-6)).view_as(mm)
                    H,W=mm.shape[-2:]; c0,c1=(ink.shape[-2]-H)//2,(ink.shape[-1]-W)//2
                    obj=ink[...,c0:c0+H,c1:c1+W].bool().expand_as(mm)
                    for r,v in (("all",mm.flatten()),("obj",mm[obj]),("bg",mm[~obj])):
                        a=acc[r]; a["n"]+=v.numel(); a["s"]+=float(v.sum()); a["s2"]+=float((v*v).sum()); a["sp"]+=int((v<SPARSE_THR).sum())
                return h
            hooks.append(mod.register_forward_hook(mk(mod)))
    m(X)
    for h in hooks: h.remove();
    del m; torch.cuda.empty_cache()
    out={}
    for r,a in (acc.items() if split else [("all",acc["all"])]):
        mean=a["s"]/a["n"]; var=a["s2"]/a["n"]-mean*mean
        out[r]=dict(mean=mean, std=float(max(var,0.0)**0.5), frac_sparse=a["sp"]/a["n"])
    return dict(**out["all"], **({"obj":out["obj"], "bg":out["bg"]} if split else {}))

R={}
for arch,ds,run in NETS:
    R[f"{arch}/{ds}"]=dict(arch=arch, **occ_stats(run, ds in GRAY))
    r=R[f"{arch}/{ds}"]; print(f"{arch}/{ds}: mean={r['mean']:.3f} std={r['std']:.3f} frac<{SPARSE_THR}={r['frac_sparse']*100:.0f}%  "+(
          f"object {r['obj']['mean']:.3f}±{r['obj']['std']:.3f} ({r['obj']['frac_sparse']*100:.0f}%)  background {r['bg']['mean']:.3f}±{r['bg']['std']:.3f} ({r['bg']['frac_sparse']*100:.0f}%)" if ds in GRAY else ""),flush=True)
json.dump(R,open(f"{ROOT}/paper_runs/_mechinterp_occ_std.json","w"),indent=2)

ORDER=[f"{a}/{d}" for a,d,_ in NETS]; x=np.arange(len(ORDER))
def bar(ax,xi,w,st,col,fs,hatch):
    lo=min(st["std"],st["mean"])                                  # m >= 0: the lower bar stops at 0
    ax.bar(xi,st["mean"],w,yerr=[[lo],[st["std"]]],color="white" if hatch else col,edgecolor=col,hatch="////" if hatch else None,
           lw=1.2,capsize=2,error_kw=dict(lw=1.0,ecolor="#333"))
    ax.text(xi,-0.07,f"{st['frac_sparse']*100:.0f}%",ha="center",fontsize=fs,color="#666")
fig,ax=plt.subplots(figsize=(12.5,5.2))
for i,k in enumerate(ORDER):
    col=ACOL[R[k]["arch"]]
    if k.split("/")[1] in GRAY:
        bar(ax,i-0.2,0.38,R[k]["obj"],col,6,False); bar(ax,i+0.2,0.38,R[k]["bg"],col,6,True)   # solid object, hatched background
    else:
        bar(ax,i,0.8,R[k],col,6.5,False)
ax.axhline(3/8,color="#2e8b57",ls="--",lw=1); ax.axhline(5/8,color="#c0392b",ls="--",lw=1)
ax.text(len(ORDER)-0.5,3/8+0.01,"survive 3/8",color="#2e8b57",fontsize=8,ha="right")
ax.text(len(ORDER)-0.5,5/8+0.01,"birth 5/8",color="#c0392b",fontsize=8,ha="right")
ax.text(-0.75,-0.07,"% <0.1:",ha="right",fontsize=6.5,color="#666")
ax.set_xticks(x); ax.set_xticklabels(ORDER,rotation=40,ha="right",fontsize=8); ax.set_ylabel("occupancy m fed to gate (mean ± std over field)",fontsize=10)
ax.set_ylim(-0.1,1.0); ax.set_xlim(-1.1,len(ORDER)-0.4)
import matplotlib.lines as ml, matplotlib.patches as mp
ax.legend([mp.Patch(color=c) for c in ACOL.values()],list(ACOL.keys()),fontsize=9,frameon=False,loc="upper center",ncol=4)

for s in ("top","right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{ROOT}/paper_figures/out/occupancy_std.png",dpi=140); print("WROTE paper_figures/out/occupancy_std.png")
