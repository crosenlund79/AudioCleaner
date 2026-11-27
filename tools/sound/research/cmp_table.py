import json, numpy as np, csv, sys
from sklearn.metrics import average_precision_score
names=[r[2] for r in list(csv.reader(open('/home/user/sed/class_labels_indices.csv')))[1:]]; L={n:i for i,n in enumerate(names)}
lab=json.load(open('/home/user/sed/labels_idx.json'))
cats=json.load(open('/home/user/AudioCleaner/assets/sound/categories.json'))['categories']
sel=json.load(open('/home/user/sed/subset_cmp.json'))
runs={}
cc=np.load('/home/user/sed/scores/clip_check.npz'); idx={v:i for i,v in enumerate(cc['sel'])}
runs['mn10 (ONNX, app)']=(cc['clip'][[idx[v] for v in sel]].astype(float), None)
for key,f in [('dymn10','cmp_dymn10'),('mn10 (torch)','cmp_mn10_torch'),('dymn20','cmp_dymn20'),('mn40_ext','cmp_mn40'),('CNN14','cmp_cnn14'),('CNN14-DecisionMax','cmp_cnn14_dlm'),('AST','cmp_ast')]:
    try:
        d=np.load(f'/home/user/sed/scores/{f}.npz')
    except FileNotFoundError:
        continue
    assert list(d['vids'])==sel
    t=None
    for k in d.files:
        if k.startswith('t_') or k=='seconds': t=float(d[k])
    runs[key]=(d['clip'].astype(float), t)
chk=["vacuum","hairdryer","microwave","blender","frying","toothbrush","tap_water","fan","tv","videogame","music","dog","laughter","knock","crying","cough","door","bang","footsteps","engine","dishes","shout","scream","sneeze"]
print("| Category | n | "+" | ".join(runs)+" |"); print("|---|---|"+"---|"*len(runs))
res={k:[] for k in runs}
for c in cats:
    if c['id'] not in chk: continue
    cols=[L[n] for n in c['labels']]
    y=np.array([any(i in cols for i in lab[v]) for v in sel])
    row=[]
    for k,(A,_) in runs.items():
        ap=average_precision_score(y, A[:,cols].max(1)); res[k].append(ap); row.append(f"{ap:.2f}")
    print(f"| {c['display']} | {y.sum()} | "+" | ".join(row)+" |")
print("| **Mean** | | "+" | ".join(f"**{np.mean(v):.3f}**" for v in res.values())+" |")
print("| s per 10 s clip (CPU, shared) | | "+" | ".join(f"{(t/len(sel)):.3f}" if t else "-" for _,t in runs.values())+" |")
