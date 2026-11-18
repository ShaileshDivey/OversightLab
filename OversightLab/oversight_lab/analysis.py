"""Summaries and manuscript-ready figure, with no claims of significance from a smoke test."""
import argparse, json
from pathlib import Path

def analyze(folder):
    import pandas as pd
    import matplotlib.pyplot as plt
    folder=Path(folder)
    rows=[json.loads(t) for t in (folder/'results.jsonl').read_text().splitlines() if t.strip()]
    df=pd.DataFrame(rows)
    if df.empty:print('No data');return
    df['model']=df.model.astype(str)
    cols=['condition','round','common_score','hidden_score','gap']
    summary=df.groupby(['condition','round'],as_index=False)[['common_score','hidden_score','gap']].agg('mean')
    summary.to_csv(folder/'summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5))
    for label,sub in summary.groupby('condition'):
        ax.plot(sub['round'],sub['gap'],marker='o',label=label)
    ax.axhline(0,linewidth=1,color='black',alpha=.5)
    ax.set(xlabel='Experience round',ylabel='Common probe − hidden quality',title='Differential evaluation performance (descriptive)')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(folder/'learning_gap.png',dpi=170)
    plt.close(fig)
    print(summary.to_string(index=False))
    if (folder/'usage.jsonl').exists():
        u=pd.DataFrame([json.loads(t) for t in (folder/'usage.jsonl').read_text().splitlines() if t.strip()])
        if not u.empty:print('\nAPI totals:',u[['input_tokens','output_tokens','estimated_usd']].sum().to_dict())
    print('\nResults:',folder.resolve())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results');analyze(p.parse_args().output)
