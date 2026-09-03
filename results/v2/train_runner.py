#!/usr/bin/env python
import os, sys, json, math, time, random, argparse, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import SiglipVisionModel, AutoImageProcessor
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm

TRAIN_DIR=Path('/workspace/dataset/SE/TRAIN')
TEST_DIR=Path('/workspace/dataset/SE/TEST')
SOLUTION_PATH=Path('/workspace/dataset/SE/TRAIN/Solution.csv')
OUT=Path('/workspace/output/siglip2_b384_condsev_4gpu')
MANIFEST=OUT/'train_manifest_5fold.csv'
MODEL_ID='google/siglip2-base-patch16-384'
SEED=20260901
IMAGE_SIZE=384
PAD_RGB=(128,128,128)
DROPOUT=0.15
UNFREEZE_LAST_N=4
CV_BATCH=32
FINAL_GLOBAL_BATCH=32
EVAL_BATCH=64
NUM_WORKERS=12
HEAD_EPOCHS=2
MAX_PARTIAL_EPOCHS=5
HEAD_LR=7e-4
PARTIAL_HEAD_LR=1e-4
PARTIAL_BACKBONE_LR=1e-5
WEIGHT_DECAY=0.05
WARMUP_RATIO=0.08
GRAD_CLIP=1.0
W_J=0.35
W_K=0.65
HFLIP_P=0.50
BRIGHTNESS=0.12
CONTRAST=0.12
SATURATION=0.08
JENIS=['BANJIR','GEMPA BUMI','KEBAKARAN']
KERUSAKAN=['KERUSAKAN RINGAN','KERUSAKAN SEDANG','KERUSAKAN BERAT']
IDX_TO_JENIS={i:x for i,x in enumerate(JENIS)}
IDX_TO_KER={i:x for i,x in enumerate(KERUSAKAN)}
SUB_JENIS={'BANJIR':1,'GEMPA BUMI':2,'KEBAKARAN':3}
# Explicit submission encoding. Keep this fixed and documented; never infer it from TEST predictions.
SUB_KER={'KERUSAKAN BERAT':1,'KERUSAKAN RINGAN':2,'KERUSAKAN SEDANG':3}

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True

def row_micro(yj,pk,yjhat,pkhat):
    yt=np.concatenate([np.asarray(yj),np.asarray(pk)])
    yp=np.concatenate([np.asarray(yjhat),np.asarray(pkhat)])
    return float(f1_score(yt,yp,average='micro'))

def metrics(yj,yk,pj,pk):
    return {'micro_f1':row_micro(yj,yk,pj,pk),
            'acc_jenis':float(accuracy_score(yj,pj)),
            'acc_kerusakan':float(accuracy_score(yk,pk))}

processor=AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
mean=np.array(processor.image_mean,dtype=np.float32).reshape(3,1,1)
std=np.array(processor.image_std,dtype=np.float32).reshape(3,1,1)

def decode(path,train=False,force_flip=False):
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert('RGB')
        if train:
            if random.random()<HFLIP_P: im=ImageOps.mirror(im)
            if BRIGHTNESS>0: im=ImageEnhance.Brightness(im).enhance(1+random.uniform(-BRIGHTNESS,BRIGHTNESS))
            if CONTRAST>0: im=ImageEnhance.Contrast(im).enhance(1+random.uniform(-CONTRAST,CONTRAST))
            if SATURATION>0: im=ImageEnhance.Color(im).enhance(1+random.uniform(-SATURATION,SATURATION))
        elif force_flip:
            im=ImageOps.mirror(im)
        w,h=im.size
        scale=min(IMAGE_SIZE/w,IMAGE_SIZE/h)
        nw=max(1,round(w*scale)); nh=max(1,round(h*scale))
        im=im.resize((nw,nh),Image.Resampling.BICUBIC)
        canvas=Image.new('RGB',(IMAGE_SIZE,IMAGE_SIZE),PAD_RGB)
        canvas.paste(im,((IMAGE_SIZE-nw)//2,(IMAGE_SIZE-nh)//2))
        arr=np.asarray(canvas,dtype=np.float32).transpose(2,0,1)/255.0
        arr=(arr-mean)/std
        return torch.from_numpy(arr)

class ImgDS(Dataset):
    def __init__(self,df,train=False,labelled=True,flip=False):
        self.df=df.reset_index(drop=True); self.train=train; self.labelled=labelled; self.flip=flip
    def __len__(self):return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; x=decode(r.path,self.train,self.flip)
        out={'pixel_values':x,'row_index':int(r.row_index) if 'row_index' in self.df.columns else i}
        if self.labelled:
            out['jenis_idx']=int(r.jenis_idx); out['kerusakan_idx']=int(r.kerusakan_idx)
        return out

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision=SiglipVisionModel.from_pretrained(MODEL_ID, local_files_only=True)
        d=self.vision.config.hidden_size
        self.feature_norm=nn.LayerNorm(d)
        self.drop=nn.Dropout(DROPOUT)
        self.disaster=nn.Linear(d,3)
        self.severity=nn.ModuleList([nn.Linear(d,3) for _ in range(3)])
    def features(self,pixel_values):
        o=self.vision(pixel_values=pixel_values,return_dict=True)
        z=o.pooler_output if getattr(o,'pooler_output',None) is not None else o.last_hidden_state.mean(1)
        return self.drop(self.feature_norm(z.float()))
    def forward(self,pixel_values):
        z=self.features(pixel_values)
        j=self.disaster(z)
        cond=torch.stack([h(z) for h in self.severity],dim=1) # B,3,3
        jp=F.softmax(j,dim=-1)
        kp=torch.sum(jp.unsqueeze(-1)*F.softmax(cond,dim=-1),dim=1)
        return j,cond,kp

def set_stage(m,stage):
    for p in m.vision.parameters():p.requires_grad=False
    for p in m.feature_norm.parameters():p.requires_grad=True
    for p in m.disaster.parameters():p.requires_grad=True
    for h in m.severity:
        for p in h.parameters():p.requires_grad=True
    if stage=='partial':
        core=getattr(m.vision,'vision_model',m.vision)
        if hasattr(core,'encoder') and hasattr(core.encoder,'layers'): layers=core.encoder.layers
        elif hasattr(m.vision,'encoder') and hasattr(m.vision.encoder,'layers'): layers=m.vision.encoder.layers
        else: raise AttributeError('Could not locate SigLIP2 transformer layers')
        for layer in layers[-UNFREEZE_LAST_N:]:
            for p in layer.parameters():p.requires_grad=True
        for attr in ('post_layernorm','head'):
            module=getattr(core,attr,None)
            if module is not None:
                for p in module.parameters():p.requires_grad=True

def make_optimizer(m,stage):
    head=[]; back=[]
    for name,p in m.named_parameters():
        if not p.requires_grad:continue
        (back if name.startswith('vision.') else head).append(p)
    groups=[]
    if back:groups.append({'params':back,'lr':PARTIAL_BACKBONE_LR if stage=='partial' else HEAD_LR})
    if head:groups.append({'params':head,'lr':PARTIAL_HEAD_LR if stage=='partial' else HEAD_LR})
    return torch.optim.AdamW(groups,weight_decay=WEIGHT_DECAY)

def cosine_sched(opt,total_steps):
    warm=max(1,int(total_steps*WARMUP_RATIO))
    def fn(step):
        if step<warm:return max(1e-8,step/warm)
        prog=(step-warm)/max(1,total_steps-warm)
        return 0.5*(1+math.cos(math.pi*min(1,prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt,fn)

def loss_fn(j,cond,yj,yk):
    lj=F.cross_entropy(j,yj)
    chosen=cond[torch.arange(len(yj),device=yj.device),yj]
    lk=F.cross_entropy(chosen,yk)
    return W_J*lj+W_K*lk

def loader(df,train,labelled,batch,sampler=None,flip=False):
    return DataLoader(ImgDS(df,train,labelled,flip),batch_size=batch,shuffle=(train and sampler is None),
        sampler=sampler,num_workers=NUM_WORKERS,pin_memory=True,persistent_workers=(NUM_WORKERS>0),drop_last=False)

def train_epoch(model,ld,opt,sch,device,distributed=False):
    model.train(); total=0.; n=0
    pbar=tqdm(ld,disable=(distributed and dist.get_rank()!=0),leave=False)
    for b in pbar:
        x=b['pixel_values'].to(device,non_blocking=True); yj=b['jenis_idx'].to(device); yk=b['kerusakan_idx'].to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            j,c,_=model(x); loss=loss_fn(j,c,yj,yk)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP); opt.step(); sch.step()
        total+=float(loss.detach())*len(x); n+=len(x)
    if distributed:
        t=torch.tensor([total,n],device=device,dtype=torch.float64); dist.all_reduce(t,op=dist.ReduceOp.SUM); total,n=t.tolist()
    return total/max(1,n)

@torch.no_grad()
def predict(model,ld,device,distributed=False):
    model.eval(); rows=[]; yj=[]; yk=[]; pj=[]; pk=[]; pjp=[]; pkp=[]
    for b in tqdm(ld,disable=(distributed and dist.get_rank()!=0),leave=False):
        x=b['pixel_values'].to(device,non_blocking=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): j,c,kprob=model(x)
        jp=F.softmax(j,dim=-1)
        rows.extend(np.asarray(b['row_index']).tolist()); pjp.append(jp.float().cpu().numpy()); pkp.append(kprob.float().cpu().numpy())
        pj.extend(jp.argmax(-1).cpu().tolist()); pk.extend(kprob.argmax(-1).cpu().tolist())
        if 'jenis_idx' in b:
            yj.extend(np.asarray(b['jenis_idx']).tolist()); yk.extend(np.asarray(b['kerusakan_idx']).tolist())
    return {'row_index':np.asarray(rows),'yj':np.asarray(yj),'yk':np.asarray(yk),'pj':np.asarray(pj),'pk':np.asarray(pk),
            'pj_prob':np.concatenate(pjp),'pk_prob':np.concatenate(pkp)}

def blend(a,b):
    return {'row_index':a['row_index'],'yj':a['yj'],'yk':a['yk'],
            'pj_prob':(a['pj_prob']+b['pj_prob'])/2,'pk_prob':(a['pk_prob']+b['pk_prob'])/2}

def fit_fold(fold):
    seed_all(SEED+fold)
    df=pd.read_csv(MANIFEST); df['row_index']=np.arange(len(df))
    tr=df[df.fold!=fold].reset_index(drop=True); va=df[df.fold==fold].reset_index(drop=True)
    device=torch.device('cuda:0')
    model=Model().to(device)
    fold_dir=OUT/f'fold_{fold}'; fold_dir.mkdir(parents=True,exist_ok=True)
    best=-1; best_meta=None; best_path=fold_dir/'best.pt'
    history=[]
    stages=[('heads',HEAD_EPOCHS),('partial',MAX_PARTIAL_EPOCHS)]
    for stage,n_epochs in stages:
        set_stage(model,stage)
        opt=make_optimizer(model,stage)
        trld=loader(tr,True,True,CV_BATCH)
        vald=loader(va,False,True,EVAL_BATCH)
        sch=cosine_sched(opt,len(trld)*n_epochs)
        for ep in range(1,n_epochs+1):
            t=time.time(); loss=train_epoch(model,trld,opt,sch,device)
            pr=predict(model,vald,device); mm=metrics(pr['yj'],pr['yk'],pr['pj'],pr['pk'])
            rec={'stage':stage,'stage_epoch':ep,'loss':loss,**mm}; history.append(rec)
            print(f'FOLD {fold} {stage} {ep}/{n_epochs} loss={loss:.4f} micro={mm["micro_f1"]:.6f} J={mm["acc_jenis"]:.5f} K={mm["acc_kerusakan"]:.5f} min={(time.time()-t)/60:.1f}',flush=True)
            if mm['micro_f1']>best:
                best=mm['micro_f1']; best_meta={'stage':stage,'stage_epoch':ep,'micro_f1':best}
                torch.save({'model':model.state_dict(),'meta':best_meta},best_path)
    ck=torch.load(best_path,map_location=device); model.load_state_dict(ck['model'])
    val_plain=loader(va,False,True,EVAL_BATCH,flip=False); val_flip=loader(va,False,True,EVAL_BATCH,flip=True)
    p0=predict(model,val_plain,device); p1=predict(model,val_flip,device); pt=blend(p0,p1)
    plain_m=metrics(p0['yj'],p0['yk'],p0['pj_prob'].argmax(1),p0['pk_prob'].argmax(1))
    tta_m=metrics(pt['yj'],pt['yk'],pt['pj_prob'].argmax(1),pt['pk_prob'].argmax(1))
    np.savez_compressed(fold_dir/'oof.npz',row_index=p0['row_index'],yj=p0['yj'],yk=p0['yk'],plain_j=p0['pj_prob'],plain_k=p0['pk_prob'],tta_j=pt['pj_prob'],tta_k=pt['pk_prob'])
    meta={'fold':fold,'best_meta':best_meta,'plain_metrics':plain_m,'tta_metrics':tta_m,'history':history}
    (fold_dir/'result.json').write_text(json.dumps(meta,indent=2))
    print('FOLD_RESULT',json.dumps({k:meta[k] for k in ['fold','best_meta','plain_metrics','tta_metrics']}),flush=True)

def smoke_test():
    seed_all(SEED)
    device=torch.device('cuda:0')
    model=Model().to(device).eval()
    x=torch.zeros(1,3,IMAGE_SIZE,IMAGE_SIZE,device=device,dtype=torch.float32)
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        z=model.features(x)
    assert z.shape==(1, model.disaster.in_features), f'Unexpected feature shape: {tuple(z.shape)}'
    core=getattr(model.vision,'vision_model',model.vision)
    layers=getattr(getattr(core,'encoder',None),'layers',None)
    assert layers is not None and len(layers)>=UNFREEZE_LAST_N, 'Could not locate expected SigLIP vision transformer layers.'
    print(f'SMOKE_OK model={MODEL_ID} feature_dim={z.shape[-1]} layers={len(layers)} device={torch.cuda.get_device_name(0)}',flush=True)

def enumerate_test():
    rows=[]
    for p in sorted(TEST_DIR.iterdir(),key=lambda x:int(x.stem) if x.stem.isdigit() else x.name):
        if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.jfif','.bmp','.webp'}:
            rows.append({'path':str(p),'id':p.stem})
    df=pd.DataFrame(rows); df['row_index']=np.arange(len(df)); return df

def init_ddp():
    dist.init_process_group('nccl')
    rank=dist.get_rank(); local=int(os.environ['LOCAL_RANK']); world=dist.get_world_size()
    torch.cuda.set_device(local); return rank,local,world,torch.device(f'cuda:{local}')

def final_ddp(selection_path):
    rank,local,world,device=init_ddp(); seed_all(SEED+999+rank)
    sel=json.loads(Path(selection_path).read_text()); head_epochs=int(sel['head_epochs']); partial_epochs=int(sel['partial_epochs']); use_tta=bool(sel['use_tta'])
    assert FINAL_GLOBAL_BATCH%world==0
    per_gpu=FINAL_GLOBAL_BATCH//world
    df=pd.read_csv(MANIFEST); df['row_index']=np.arange(len(df))
    model=Model().to(device)
    stages=[('heads',head_epochs)] + ([('partial',partial_epochs)] if partial_epochs>0 else [])
    for stage,n_epochs in stages:
        set_stage(model,stage)
        # Re-wrap at each stage because trainable parameter set changes.
        ddp=DDP(model,device_ids=[local],output_device=local,broadcast_buffers=False,find_unused_parameters=True)
        sampler=DistributedSampler(ImgDS(df,True,True),num_replicas=world,rank=rank,shuffle=True,seed=SEED,drop_last=False)
        # construct loader from same dataset used by sampler
        ds=sampler.dataset
        ld=DataLoader(ds,batch_size=per_gpu,sampler=sampler,num_workers=NUM_WORKERS,pin_memory=True,persistent_workers=True)
        opt=make_optimizer(model,stage); sch=cosine_sched(opt,len(ld)*n_epochs)
        for ep in range(1,n_epochs+1):
            sampler.set_epoch(ep + (0 if stage=='heads' else 100))
            t=time.time(); loss=train_epoch(ddp,ld,opt,sch,device,distributed=True)
            if rank==0: print(f'FULL_DDP {stage} {ep}/{n_epochs} loss={loss:.4f} min={(time.time()-t)/60:.1f}',flush=True)
        del ddp; torch.cuda.empty_cache(); dist.barrier()
    final_path=OUT/'full_train_final.pt'
    if rank==0:
        torch.save({'model':model.state_dict(),'selection':sel},final_path)
        print('Saved',final_path,flush=True)
    dist.barrier()
    if rank==0:
        # TEST inference on rank 0 only after final TRAIN completes.
        test=enumerate_test(); l0=loader(test,False,False,EVAL_BATCH,flip=False); p0=predict(model,l0,device)
        probs_j=p0['pj_prob']; probs_k=p0['pk_prob']
        if use_tta:
            l1=loader(test,False,False,EVAL_BATCH,flip=True); p1=predict(model,l1,device)
            probs_j=(probs_j+p1['pj_prob'])/2; probs_k=(probs_k+p1['pk_prob'])/2
        pred_j=probs_j.argmax(1); pred_k=probs_k.argmax(1)
        # Build rows from sample solution order, preserving ID strings exactly.
        sol=pd.read_csv(SOLUTION_PATH,sep=None,engine='python')
        out=sol.copy(); target=[]
        by_id={str(test.iloc[i].id):(SUB_JENIS[IDX_TO_JENIS[int(pred_j[i])]],SUB_KER[IDX_TO_KER[int(pred_k[i])]]) for i in range(len(test))}
        for rid in out['ID'].astype(str):
            base,suffix=rid.rsplit('_',1)
            assert base in by_id, f'Missing test ID {base}'
            target.append(by_id[base][0] if suffix=='jenis' else by_id[base][1])
        out['Target']=target
        out.to_csv(OUT/'submission.csv',index=False)
        np.savez_compressed(OUT/'test_probabilities.npz',ids=test.id.astype(str).to_numpy(),jenis_prob=probs_j,kerusakan_prob=probs_k)
        print(out.head(10).to_string(index=False)); print('Saved',OUT/'submission.csv',flush=True)
    dist.barrier(); dist.destroy_process_group()

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['smoke','fold','final'],required=True); ap.add_argument('--fold',type=int); ap.add_argument('--selection')
    a=ap.parse_args()
    if a.mode=='smoke':
        smoke_test()
    elif a.mode=='fold':
        assert a.fold is not None; fit_fold(a.fold)
    else:
        assert a.selection; final_ddp(a.selection)
