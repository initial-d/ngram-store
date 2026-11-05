"""Plot all news-task controls from the validated numeric summary (Matplotlib)."""
import argparse
from pathlib import Path

from .analyze_news_classifier import read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    cfg = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = read(cfg.summary)
    cfg.output.mkdir(parents=True,exist_ok=True)
    for task,record in data['tasks'].items():
        rows = record['rows']
        names = [r['backend'] for r in rows]
        colors = ['#13795b' if n.startswith('ngram-') else '#586a88' for n in names]
        fig, axes = plt.subplots(1,3,figsize=(15,8),sharey=True)
        for ax,field,title,unit in zip(axes,
                ('model_mib','peak_rss_mib','batch32_docs_per_second'),
                ('Complete model files','Observed test peak RSS','Complete request throughput'),
                ('MiB','MiB','documents / second')):
            values = [r[field] for r in rows]
            ax.barh(range(len(rows)),values,color=colors,height=.65)
            ax.set_xlabel(unit)
            ax.set_title(title,fontsize=11)
            ax.set_xlim(0,max(values)*1.32)
            ax.grid(axis='x',alpha=.2)
            ax.set_axisbelow(True)
            for i,value in enumerate(values):
                label = f'{value:,.0f}' if field=='batch32_docs_per_second' else f'{value:.2f}'
                ax.text(value+max(values)*.02,i,label,va='center',fontsize=8)
            ax.spines[['top','right']].set_visible(False)
        axes[0].set_yticks(range(len(rows)),[f"{r['backend']}  |  {r['accuracy']*100:.2f}%" for r in rows],fontsize=9)
        axes[0].invert_yaxis()
        fig.suptitle(f'AG News: {"nonlinear deep averaging" if task=="relu" else "linear averaging (folding control retained)"}',fontsize=15)
        fig.text(.015,.02,'Three-seed medians. Labels: accuracy on all 7,600 official test documents. Throughput: 512 validation requests, batch 32, 5 repetitions.\n'
            'Shared warm CPU, one query thread; no Torch inference. RSS includes runtime, keys, request buffers and scratch; no enforced memory cap.\n'
            'Precomputed words discard OOV subword features. NGramStore aggregate projection and INT8 change rows; bounded FP32 has a separate source-prediction certificate.',fontsize=9)
        fig.tight_layout(rect=(0,.11,1,.95))
        fig.savefig(cfg.output/f'{task}.png',dpi=160)
        fig.savefig(cfg.output/f'{task}.svg')
        plt.close(fig)


if __name__=='__main__': main()
