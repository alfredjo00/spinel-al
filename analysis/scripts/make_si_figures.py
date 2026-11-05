#!/usr/bin/env python3
"""Supplementary Figures S3 to S6 of the article.

  figS03_dynamics         trailing five-cycle mean OTC and the acquired fraction above each
                          OTC threshold, per surrogate
  figS04_selected_scores  selected acquisition scores relative to the structure ceiling
  figS05_chemistry_joint  configurational entropy and OTC of the acquisitions of each
                          strategy against the initial dataset
  figS06_reactor_offset   error of the fuel-reactor reconstruction of the vacancy-input
                          target and its effect on OTC

Inputs: acquisitions.csv, initial_dataset.csv, run_metrics.csv, cycle_metrics.csv,
selected_scores_all.csv and reactor_offset_audit.csv. Outputs: the PDFs in figures/.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import pandas as pd

from paths import DATA, FIGURES as OUT

WIDTH = 180/25.4
FAMILIES = {'DE':['de','de_clip','de_land','dde'],
            'GP':['gp','gp_clip','mogp','dgp'], 'MC':['mc','mc_clip']}
COLORS = {'DE':'#D55E00', 'GP':'#0072B2', 'MC':'#009E73'}
REPS = ['Scalar','Clipped scalar','Gibbs vector','Vacancy input']
SHORT = ['Scalar','Clipped','Vector','Vacancy']
STYLES = ['-', '--', '-.', ':']
MARKERS = ['o','s','^','D']
ARMS = sum(FAMILIES.values(), [])
META = {arm:(family,i) for family,arms in FAMILIES.items() for i,arm in enumerate(arms)}
LABELS = [f'{META[a][0]} · {REPS[META[a][1]]}' for a in ARMS]
INPUTS = ['acquisitions.csv','initial_dataset.csv','run_metrics.csv',
          'cycle_metrics.csv','selected_scores_all.csv','reactor_offset_audit.csv']


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
        'axes.labelsize':8, 'axes.titlesize':9, 'axes.titleweight':'bold',
        'xtick.labelsize':7.5, 'ytick.labelsize':7.5, 'legend.fontsize':7.5,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.65,
        'axes.edgecolor':'#666666', 'xtick.major.size':3, 'ytick.major.size':3,
        'grid.color':'#E7E7E7', 'grid.linewidth':.45, 'lines.linewidth':1.5,
        'pdf.fonttype':42, 'svg.fonttype':'none', 'mathtext.fontset':'dejavusans',
        'savefig.facecolor':'white', 'figure.facecolor':'white'})


def panel(ax, letter, title='', grid='y'):
    ax.set_title(f'{letter}   {title}'.rstrip(), loc='left', pad=7)
    if grid:
        ax.grid(axis=grid)
    ax.set_axisbelow(True)


def reps_legend(fig, y=.01):
    fig.legend(handles=[Line2D([],[],color='#333333',ls=STYLES[i],marker=MARKERS[i],
                              ms=3.5,label=rep) for i,rep in enumerate(REPS)],
               loc='lower center', bbox_to_anchor=(.53,y), ncol=4,
               columnspacing=1.3, handlelength=2.8, frameon=False)


def save(fig, name):
    # Fixed canvas preserves the authored 180 mm width and actual font sizes.
    canvas=FigureCanvasAgg(fig);canvas.draw()
    bounds=fig.get_tightbbox(canvas.get_renderer())
    w,h=fig.get_size_inches()
    fits=bounds.x0>=0 and bounds.y0>=0 and bounds.x1<=w and bounds.y1<=h
    assert fits, (name, bounds.extents, w, h)
    fig.savefig(OUT/f'{name}.pdf', dpi=300)
    print(name, flush=True)


def clean_rows(ax, labels=LABELS):
    ax.set_ylim(len(labels)-.5,-.5)
    ax.set_yticks(range(len(labels)),labels)
    ax.tick_params(axis='y',length=0)
    ax.spines['left'].set_visible(False)
    for y in [3.5,7.5]:
        ax.axhline(y,color='#DDDDDD',lw=.6,zorder=0)


def supporting(acq,initial,scores,offsets):
    fig,axes=plt.subplots(2,3,figsize=(WIDTH,4.9))
    fig.subplots_adjust(left=.105,right=.98,top=.93,bottom=.16,hspace=.43,wspace=.24)
    for k,(family,arms) in enumerate(FAMILIES.items()):
        for arm in arms:
            _,i=META[arm];curves=[];tails=[];threshold=np.linspace(4,6.5,501)
            for _,g in acq[acq.arm==arm].groupby('replicate'):
                b=g.groupby('cycle').otc.agg(['sum','count'])
                y=(b['sum'].rolling(5).sum()/b['count'].rolling(5).sum()).iloc[4:]
                curves.append(y.to_numpy());t=(g.otc.to_numpy()[:,None]>=threshold).mean(0);tails.append(t)
                axes[0,k].plot(y.index,y,color=COLORS[family],ls=STYLES[i],lw=.65,alpha=.3)
                axes[1,k].plot(threshold,t,color=COLORS[family],ls=STYLES[i],lw=.65,alpha=.3)
            axes[0,k].plot(np.arange(5,21),np.mean(curves,0),color=COLORS[family],ls=STYLES[i],marker=MARKERS[i],markevery=5,ms=3)
            axes[1,k].plot(threshold,np.mean(tails,0),color=COLORS[family],ls=STYLES[i],marker=MARKERS[i],markevery=100,ms=3)
        axes[0,k].set(xlim=(4.9,20.1),ylim=(3.8,5.9),xticks=[5,10,15,20],xlabel='Acquisition cycle')
        axes[1,k].set(xlim=(3.98,6.52),ylim=(-.02,1.02),xticks=[4,5,6],xlabel='OTC threshold (wt.%)')
        for row in range(2):panel(axes[row,k],chr(97+row*3+k),family)
        if k==0:
            axes[0,k].set_ylabel('Trailing 5-cycle OTC (wt.%)');axes[1,k].set_ylabel('Acquired fraction ≥ threshold')
        else:
            axes[0,k].tick_params(labelleft=False);axes[1,k].tick_params(labelleft=False)
    reps_legend(fig,.015);save(fig,'figS03_dynamics')

    fig,axes=plt.subplots(1,3,figsize=(WIDTH,2.7))
    fig.subplots_adjust(left=.10,right=.98,top=.88,bottom=.28,wspace=.20)
    for k,family in enumerate(FAMILIES):
        ax=axes[k]
        for i,arm in enumerate(FAMILIES[family]):
            s=scores[scores.arm==arm];xx=s.score/s.cap
            # Scores flagged as at the ceiling (at_cap, within 1e-5) are drawn at exactly one.
            xx=xx.mask(s.at_cap,1)
            x=np.sort(xx.to_numpy()); y=np.arange(1,len(x)+1)/len(x)
            assert x.min()>=.75 and x.max()<=1.75
            ax.step(np.r_[.75,x,1.75],np.r_[0,y,1],where='post',color=COLORS[family],ls=STYLES[i],lw=1.5)
        ax.axvline(1,color='#777777',ls=':',lw=.7)
        ax.set(xlim=(.75,1.75),ylim=(-.02,1.03),xticks=[.8,1,1.2,1.4,1.6],xlabel='Selected score / structure ceiling')
        panel(ax,chr(97+k),family)
        if k==0:ax.set_ylabel('Cumulative fraction')
        else:ax.tick_params(labelleft=False)
    reps_legend(fig,.015);save(fig,'figS04_selected_scores')

    fig=plt.figure(figsize=(WIDTH,7.2))
    gs=fig.add_gridspec(3,4,left=.09,right=.98,bottom=.07,top=.95,hspace=.5,wspace=.32)
    for row,(family,arms) in enumerate(FAMILIES.items()):
        for i,arm in enumerate(arms):
            cell=gs[row,i].subgridspec(2,2,height_ratios=[.23,1],width_ratios=[1,.22],hspace=.03,wspace=.03)
            ax=fig.add_subplot(cell[1,0]); top=fig.add_subplot(cell[0,0],sharex=ax);right=fig.add_subplot(cell[1,1],sharey=ax)
            a=acq[acq.arm==arm]
            ax.scatter(initial.entropy,initial.otc,color='#999999',s=2,alpha=.13,lw=0,rasterized=True)
            ax.scatter(a.entropy,a.otc,color=COLORS[family],s=3,alpha=.25,lw=0,rasterized=True)
            for g,c in [(initial,'#AAAAAA'),(a,COLORS[family])]:
                top.hist(g.entropy,bins=np.linspace(0,4.5,24),density=True,histtype='step',color=c,lw=.85)
                right.hist(g.otc,bins=np.linspace(-.001,6.6,30),density=True,histtype='step',color=c,lw=.85,orientation='horizontal')
            ax.set(xlim=(0,4.5),ylim=(-.1,6.6),xticks=[0,2,4],yticks=[0,3,6])
            top.axis('off');right.axis('off');top.set_title(f'{chr(97+ARMS.index(arm))}  {family} · {SHORT[i]}',loc='left',fontsize=8)
            if i==0:ax.set_ylabel('OTC (wt.%)')
            else:ax.tick_params(labelleft=False)
            ax.set_xlabel(r'$S_{\mathrm{conf}}/R$')
    note=fig.add_subplot(gs[2,2:]);note.axis('off')
    note.text(0,1,'Acquired records and marginal distributions\n\nGray: common initial set\nColor: acquired compositions\n\nEach marginal distribution is normalized\nseparately. Five runs per strategy.',va='top',fontsize=8,linespacing=1.7)
    save(fig,'figS05_chemistry_joint')

    fig,axes=plt.subplots(1,2,figsize=(WIDTH,3.8))
    fig.subplots_adjust(left=.235,right=.98,top=.91,bottom=.16,wspace=.24)
    acquired=offsets[offsets.subset=='acquired']
    for k,key in enumerate(['fr_energy_rmse_eV','otc_offset_mae_pp']):
        for j,arm in enumerate(ARMS):
            f,i=META[arm];v=acquired[acquired.arm==arm][key].to_numpy()
            axes[k].scatter(v,j+np.linspace(-.26,.26,5),color=COLORS[f],marker=MARKERS[i],s=14,lw=0)
        clean_rows(axes[k]);panel(axes[k],chr(97+k),['FR reconstruction error','Effect on evaluated OTC'][k],grid='x')
        if k:axes[k].set_yticklabels([])
    axes[0].set_xlabel('Energy RMSE (eV)');axes[1].set_xlabel('OTC MAE (percentage points)')
    save(fig,'figS06_reactor_offset')



def main():
    OUT.mkdir(parents=True,exist_ok=True);style()
    acq,initial,runs,cycles,scores,offsets=[pd.read_csv(DATA/f) for f in INPUTS]
    assert len(acq)==9982 and len(runs)==50 and len(cycles)==1000
    assert len(scores)==10000 and (scores.groupby(['arm','replicate']).size()==200).all()
    assert set(acq.arm)==set(ARMS)
    assert np.allclose(acq.filter(regex='^count_').sum(axis=1),36)
    for (arm,rep),g in acq.groupby(['arm','replicate']):
        r=runs[(runs.arm==arm)&(runs.replicate==rep)].iloc[0]
        assert np.isclose(g.otc.mean(),r.mean_otc) and np.isclose(g.otc.max(),r.max_otc)
        assert np.isclose(np.isclose(g.vacancy_diff,7,atol=1e-5).mean(),r.saturated_fraction)
    supporting(acq,initial,scores,offsets)
    plt.close('all')


if __name__=='__main__':main()
