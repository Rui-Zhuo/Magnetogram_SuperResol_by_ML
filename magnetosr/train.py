"""Train the published architectures with shared splits and the original loss."""
import argparse
import copy
import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from .models import make
from .coordinates import make_coord
from .inference import read_sample


class PairedDataset(Dataset):
    def __init__(self, root, names, augment=False):
        self.root, self.names, self.augment = root, names, augment
    def __len__(self): return len(self.names)
    def __getitem__(self, index):
        hmi, radius, sp = read_sample(self.root/self.names[index])
        if sp is None or sp.shape != (312,336) or hmi.shape != (200,200) or not np.isfinite(sp).all():
            raise ValueError(f'{self.names[index]}: require finite 200x200 -> 312x336 training pair')
        inp = np.stack([hmi, radius])
        # Preserve the original gate: 50% chance to consider independent flips.
        if self.augment and random.random() < .5:
            for axis in [-1, -2]:
                if random.random() < .5:
                    inp, sp = np.flip(inp, axis), np.flip(sp, axis)
        return torch.from_numpy(inp.copy())/200., torch.from_numpy(sp.copy()).reshape(-1,1)/200.


def combined_loss(pred, gt, weight):
    """(1-w)*MAE + w*(1-Pearson CC), normalized fields, batch mean."""
    p, g = pred.flatten(1), gt.flatten(1)
    p, g = p-p.mean(1,keepdim=True), g-g.mean(1,keepdim=True)
    cc = (p*g).sum(1) / torch.sqrt((p.square().sum(1)+1e-8)*(g.square().sum(1)+1e-8))
    cc = cc.clamp(-.999,.999)
    return (1-weight)*(pred-gt).abs().mean() + weight*(1-cc).mean()


def run_epoch(model, loader, device, weight, optimizer=None):
    model.train(optimizer is not None)
    total, count = 0., 0
    for inp, gt in loader:
        inp, gt = inp.to(device), gt.to(device)
        coord = make_coord((312,336)).to(device)[None].expand(inp.shape[0],-1,-1)
        cell = torch.ones_like(coord); cell[...,0]*=2/312; cell[...,1]*=2/336
        with torch.set_grad_enabled(optimizer is not None):
            pred = model(inp,coord,cell)
            loss = combined_loss(pred, gt, weight)
            if not torch.isfinite(loss): raise ValueError('Non-finite loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        total += loss.item()*inp.shape[0]; count += inp.shape[0]
    return total/count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/train_pm_lte.yaml'))
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--split', type=Path, default=Path('data/splits/dataset_split_10115.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', type=Path)
    a = p.parse_args(); cfg = yaml.safe_load(a.config.read_text()); splits = json.loads(a.split.read_text())
    for key in ['train','val','test']:
        if not splits.get(key): raise ValueError(f'Empty {key} split')
    all_names = sum([splits[k] for k in ['train','val','test']], [])
    if len(all_names) != len(set(all_names)): raise ValueError('Split overlap/duplicate detected')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    loaders = {k: DataLoader(PairedDataset(a.data,splits[k],k=='train'),batch_size=cfg['batch_size'],
                             shuffle=k=='train',num_workers=0) for k in ['train','val']}
    model = make(cfg['model']).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,cfg['milestones'],gamma=.5)
    best, start = float('inf'), 1
    if a.resume:
        checkpoint=torch.load(a.resume,map_location=a.device,weights_only=True)
        model.load_state_dict(checkpoint['model']['sd']); optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']); best=checkpoint['best']; start=checkpoint['epoch']+1
    for epoch in range(start,cfg['epochs']+1):
        train_loss=run_epoch(model,loaders['train'],a.device,cfg['cc_weight'],optimizer)
        val_loss=run_epoch(model,loaders['val'],a.device,cfg['cc_weight'])
        scheduler.step()
        improved=val_loss<best; best=min(best,val_loss)
        spec=copy.deepcopy(cfg['model']); spec['sd']=model.state_dict()
        state=dict(model=spec,epoch=epoch,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),best=best)
        torch.save(state,a.output/'last.pth')
        if improved: torch.save(dict(model=spec,epoch=epoch,normalization=200.),a.output/'best.pth')
        with (a.output/'loss.jsonl').open('a') as f: f.write(json.dumps(dict(epoch=epoch,train=train_loss,val=val_loss))+'\n')
        print(f'Epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}',flush=True)


if __name__ == '__main__': main()
