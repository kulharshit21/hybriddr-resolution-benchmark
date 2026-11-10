"""Summarize development-only loss comparisons, including grade-1 trade-offs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import read_json,save_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    config=read_json(args.out/'STUDY_CONTRACT.json')['config']
    rows=[]
    for recipe in config['recipes']:
        for seed in config['seeds']:
            name=f"{recipe['name']}_seed{seed}";run=args.out/'runs'/name
            history=read_json(run/'history.json');best=max(history,key=lambda x:x['mean_source_val_qwk'])
            saved=np.load(run/'best_validation_private.npz',allow_pickle=False)
            if int(saved['epoch'])!=best['epoch']:raise ValueError('Best validation cache epoch mismatch')
            for source,grade in best['validation_per_grade'].items():
                values=best['validation'][source]
                rows.append(dict(recipe=recipe['name'],seed=seed,source=source,best_epoch=best['epoch']+1,
                    mean_source_qwk=best['mean_source_val_qwk'],qwk=values['qwk'],macro_f1=values['macro_f1'],
                    balanced_accuracy=values['balanced_accuracy'],grade1_recall=grade['recall'][1],grade1_precision=grade['precision'][1],
                    grade1_f1=grade['f1'][1],grade1_support=grade['support'][1],grade1_prediction_count=grade['prediction_counts'][1],
                    ece=values['ece_15_equal_width'],mae=values['mae']))
    frame=pd.DataFrame(rows);frame.to_csv(args.out/'validation_results_by_seed.csv',index=False)
    frame.groupby(['recipe','source'])[['qwk','macro_f1','balanced_accuracy','grade1_recall','grade1_f1','mae','ece']].agg(['mean','std','count']).to_csv(args.out/'validation_results_summary.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,metric in zip(axes,['qwk','grade1_recall']):
        for i,recipe in enumerate(config['recipes']):
            values=frame[frame.recipe==recipe['name']].groupby('seed')[metric].mean()
            ax.scatter(np.full(len(values),i)+np.linspace(-.08,.08,len(values)),values,label=recipe['name'])
            ax.plot([i],[values.mean()],'kx')
        ax.set(xticks=range(5),xticklabels=[r['name'].replace('aligned_','') for r in config['recipes']],ylabel='Source-mean validation '+metric,ylim=(0,1))
        ax.tick_params(axis='x',rotation=35)
    fig.suptitle('Exploratory validation at QWK-selected checkpoints; two seeds, no held-out testing')
    fig.tight_layout()
    for suffix in ['pdf','png']:fig.savefig(args.out/f'validation_tradeoffs.{suffix}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    save_json(args.out/'VALIDATION_REPORT_COMPLETE.json',dict(rows=len(frame),scope='development-only; no automatic promotion or held-out testing'))
    print(frame.groupby(['recipe','source'])[['qwk','grade1_recall','grade1_f1']].mean().to_string(),flush=True)


if __name__=='__main__':main()
