"""All-control resource plots and paired task-quality intervals (Matplotlib)."""
import argparse
from pathlib import Path

from .analyze_news_classifier import read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    cfg=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=read(cfg.summary);cfg.output.mkdir(parents=True,exist_ok=True)
    order=['source-resident','source-mmap','fp16','int8','faiss-sq8','faiss-sq4','faiss-pq-8x8',
           'faiss-cluster-8','ngram-bounded','ngram-int8','precomputed-words','precomputed-hybrid-source',
           'precomputed-hybrid-int8','precomputed-hybrid-faiss-sq4','precomputed-hybrid-ngram-int8']
    lookup={r['backend']:r for r in data['rows']};rows=[lookup[k] for k in order]
    colors=['#16785d' if 'ngram-' in r['backend'] else '#a47831' if r['backend']=='precomputed-words'
            else '#536d91' for r in rows]
    fig,axes=plt.subplots(1,3,figsize=(16,9),sharey=True)
    for ax,field,title,unit in zip(axes,('model_mib','peak_rss_mib','batch32_docs_per_second'),
            ('Complete model files','Observed evaluation peak RSS','Complete-text throughput'),
            ('MiB','MiB','docs/s: calibration, batch 32')):
        values=[r[field] for r in rows];maximum=max(values)
        ax.barh(range(len(rows)),values,color=colors,height=.65)
        ax.set_xlabel(unit);ax.set_title(title,fontsize=11);ax.set_xlim(0,maximum*1.3)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
        for i,value in enumerate(values):
            ax.text(value+maximum*.02,i,f'{value:,.0f}' if field.endswith('second') else f'{value:.2f}',va='center',fontsize=8)
    axes[0].set_yticks(range(len(rows)),[f"{r['backend']} | {r['accuracy']*100:.2f}%" for r in rows],fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle('Independent HuffPost news: frozen AG News classifiers, all 15 controls',fontsize=15)
    fig.text(.015,.02,'Three-seed medians. Accuracy: all 13,134 evaluation articles; calibration: 3,281 stratified articles (not a chronological split).\n'
        'Apple M5 CPU, one query worker and explicit runtime thread limits; no Torch inference, enforced RAM cap, exclusive core or cold-cache claim.\n'
        'All hybrids share the same precomputed FP32 training-word features. NGramStore gains concern the fallback table; projection/INT8 change source rows.\n'
        'Storage, source agreement and task quality are separate; all controls, including zero-OOV precomputed words, remain in the comparison.',fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.95))
    for suffix in ('png','svg'):fig.savefig(cfg.output/('resources.'+suffix),dpi=160)
    plt.close(fig)
    names=['source-mmap','precomputed-hybrid-source','precomputed-hybrid-int8',
           'precomputed-hybrid-faiss-sq4','precomputed-hybrid-ngram-int8']
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),sharey=True)
    for ax,slice_name,title in zip(axes,('all','oov_heavy'),('All 13,134 evaluation articles','1,132 naturally OOV-heavy articles')):
        for i,name in enumerate(names):
            record=data['paired_accuracy']['zero_oov_comparisons'][name][slice_name]
            value=record['difference_pp'];low,high=record['ci95_pp']
            ax.errorbar(value,i,xerr=[[max(0,value-low)],[max(0,high-value)]],fmt='o',capsize=4,
                        color='#16785d' if 'ngram-' in name else '#536d91')
        ax.axvline(0,color='#555',linestyle='--',linewidth=1)
        ax.set_title(title,fontsize=11);ax.set_xlabel('Accuracy change vs zero-OOV precomputed words (pp)')
        ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(range(len(names)),names,fontsize=9);axes[0].invert_yaxis()
    fig.suptitle('Paired task-accuracy effect of retaining subwords',fontsize=14)
    fig.text(.015,.025,'Mean paired differences across the same three fitted models; 95% document-bootstrap intervals (2,000 resamples).\n'
        'Each document is one resampled unit. This does not estimate uncertainty over all possible training seeds. All unfavorable comparisons are retained.',fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.92))
    for suffix in ('png','svg'):fig.savefig(cfg.output/('paired-quality.'+suffix),dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
