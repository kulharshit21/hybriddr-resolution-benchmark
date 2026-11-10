"""Exercise the complete explanation path for both control architectures."""
import argparse,hashlib,json,sys,tempfile
from pathlib import Path
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True);a=ap.parse_args()
    sys.path.insert(0,str(a.baseline.resolve()))
    import torch
    torch.set_num_threads(2)
    import numpy as np
    import pandas as pd
    from PIL import Image
    from models import HybridDR
    from postprocess_controls import explain
    for arch in ('cnn','gated'):
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);run=p/'run';run.mkdir();out=p/'out';out.mkdir()
            Image.fromarray(np.random.default_rng(42).integers(0,256,(56,56,3),dtype=np.uint8)).save(p/'synthetic.png')
            cfg=dict(tiny=True,pretrained=False,gradient_checkpointing=False,image_size=56,fusion_width=16,fusion_heads=2)
            recipe=dict(name=arch+'_ce',architecture=arch,loss='ce');model=HybridDR(cfg,recipe,initialize=False).eval()
            from torchvision.transforms.functional import to_tensor
            with Image.open(p/'synthetic.png') as im:tensor=to_tensor(im)
            with torch.no_grad():probs=model(tensor[None]).softmax(1).numpy()[0]
            frame=pd.DataFrame([dict(path='synthetic.png',cache='synthetic.png',source='SYNTHETIC',patient='fixture',label=1)])
            pred=frame[['path','source','patient','label']].copy();pred['pred']=int(probs.argmax());pred['calibrated_pred']=int(probs.argmax())
            for k in range(5):pred[f'p{k}']=probs[k];pred[f'cal_p{k}']=probs[k]
            pred.to_csv(run/'SYNTHETIC_internal_predictions.csv',index=False)
            torch.save(dict(config=cfg,recipe=recipe,model=model.state_dict()),run/'best.pt')
            ev=dict(checkpoint_sha256=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest())
            explain(run,ev,{'SYNTHETIC_internal':frame},p,out,'cpu',1)
            ledger=pd.read_csv(out/'case_ledger_private.csv')
            assert len(ledger)==1 and len(list(out.glob('*.png')))==1
            heat=np.load(next(out.glob('*.npy')));assert np.isfinite(heat).all() and heat.ndim==2
            print('CONTROL_ATTRIBUTION_CPU_SMOKE_PASS',arch,flush=True)
if __name__=='__main__':main()
