#!/usr/bin/env python3
"""Plot image metrics and phase-zero ROI/full axial montages across gate counts.

Pass one or more output directories created by evaluate.py for the same gating mode.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METHODS = [('OSEM','#3B6FB6'), ('BSREM + NLM','#009E73'), ('TD-DIP','#D55E00')]
METRICS = [('cnr','CNR'), ('wall_cov_percent','Wall CoV (%)'), ('ssim','SSIM'), ('psnr_db','PSNR (dB)')]


def save(fig, output, stem):
    for extension in ('png','pdf'):
        fig.savefig(output / f'{stem}.{extension}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', nargs='+', type=Path, help='Per-gate output directories from evaluate.py')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kind', choices=('all', 'metrics', 'slices'), default='all')
    p.add_argument('--display-max', type=float, default=100, help='Upper grayscale activity limit in kBq/mL')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    records=[]; panels={}; modes=set()
    for path in args.results:
        meta=json.loads((path/'result.json').read_text()); gate=int(meta['gates']); modes.add(meta['mode'])
        if gate in panels: raise ValueError(f'Duplicate {gate}-gate result')
        with (path/'image_metrics_summary.csv').open(newline='') as f: records.extend(csv.DictReader(f))
        with np.load(path/'phase0_slices.npz') as a: panels[gate]={k:a[k].copy() for k in a.files}
    if len(modes)!=1: raise ValueError('Plot static and dynamic studies separately')
    gates=sorted(panels)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'figure.facecolor':'white'})
    if args.kind in ('all', 'metrics'):
        for metric,label in METRICS:
            fig,ax=plt.subplots(figsize=(7.2,4.8))
            for method,color in METHODS:
                chosen=[(i,r) for i,n in enumerate(gates) for r in records
                        if r['method']==method and int(r['gate_count'])==n and r['metric']==metric]
                if not chosen: continue
                x=[i for i,_ in chosen]; y=[float(r['median']) for _,r in chosen]
                ax.plot(x,y,marker='o',lw=2,color=color,label=method)
                ax.fill_between(x,[float(r['p2_5']) for _,r in chosen],
                               [float(r['p97_5']) for _,r in chosen],color=color,alpha=.2,linewidth=0)
            ax.set_xticks(range(len(gates)),gates);ax.set_xlabel('Gate count');ax.set_ylabel(label)
            if metric=='ssim':ax.set_ylim(0,1)
            ax.grid(axis='y',linestyle='--',alpha=.3);ax.legend(frameon=False);fig.tight_layout()
            save(fig,args.output,metric)
    if args.kind in ('all', 'slices'):
        methods=[name for name,_ in METHODS if any(f'{name}_roi' in panels[n] for n in gates)]+['Ground truth']
        for scope in ('roi','full'):
            fig,axes=plt.subplots(len(methods),len(gates),figsize=(2.3*len(gates)+1,2.3*len(methods)),squeeze=False)
            fig.subplots_adjust(left=.09,right=.998,bottom=.01,top=.93,wspace=.035,hspace=.035)
            for row,method in enumerate(methods):
                for col,gate in enumerate(gates):
                    ax=axes[row,col]; key=f'{method}_{scope}'
                    if key in panels[gate]:
                        image=np.rot90(np.flipud(panels[gate][key]),k=-1)
                        ax.imshow(image,cmap='gray',vmin=0,vmax=args.display_max,interpolation='nearest')
                    else:
                        ax.set_facecolor('black');ax.text(.5,.5,'Unavailable',color='white',ha='center',va='center',transform=ax.transAxes)
                    ax.set_xticks([]);ax.set_yticks([])
                    if row==0:ax.set_title(f'{gate} gates',pad=8)
                    if col==0:ax.set_ylabel(method)
                    for spine in ax.spines.values():spine.set_visible(False)
            save(fig,args.output,f'phase0_axial_{scope}')


if __name__=='__main__':main()
