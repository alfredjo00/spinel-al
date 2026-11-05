#!/usr/bin/env python3
"""Figures 1 to 5 and Supplementary Figures S1 and S2 of the article.

  fig01_design           the active learning protocol (a) and the ten strategies (b)
  fig02_target           one evaluated composition (a) and what each target predicts (b)
  fig03_discovery        discovery at a fixed budget (drawn in figure_common.py)
  fig04_mechanism        selected scores, ceilings and chemistry (figure_common.py)
  fig05_surrogate_value  run mean OTC against run mean ceiling for the 50 runs, the
                         ceiling-only runs and the random reference (a), Spearman
                         correlation of the held-out predictions of the eight cycle-0
                         surrogates on the random reference and on the 1,963 MC-acquired
                         compositions (b), and predicted OTC / C for the MC-acquired
                         compositions at the ceiling (c)
  figS01_individual_runs best OTC so far in each run, with the ceiling-only runs
  figS02_run_endpoints   run outcomes, with the ceiling-only runs

Inputs: acquisitions.csv, initial_dataset.csv, run_metrics.csv, cycle_metrics.csv,
selected_scores_all.csv, ceiling_run_metrics.csv, ceiling_cycle_metrics.csv,
random_baseline_additional.csv, the held-out test in holdout_test/ (made with the active
learning code, not public), and spinel_rev_mogp_s5/training.db for Figure 2a.
Outputs: the PDFs in figures/.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import figure_common as fc
from figure_common import (ARMS, COLORS, FAMILIES, INK, MARKERS, META, MUTED, OUT, REPS,
                           SHORT, WIDTH, family_brace, frame_panels, panel, row_axis, save)
from paths import DATA

HOLDOUT = DATA / 'holdout_test'
CEIL_COLOR = '#3A3A3A'
RANDOM_COLOR = '#9A9A9A'

# Held-out models in the row order of the strategies they stand for. Capped scalar
# models stand for the clipped strategies (capped after prediction).
OFFLINE = {'de': 'de', 'de_clip': 'de_cap', 'de_land': 'de_land', 'dde': 'dde',
           'gp': 'gp', 'gp_clip': 'gp_cap', 'mogp': 'mogp', 'dgp': 'dgp'}
OFFLINE_ARMS = list(OFFLINE)
ROW_LIM = (7.5, -1.4)  # extra headroom above the first row for the panel b key


def load_offline():
    metrics = pd.read_csv(HOLDOUT / 'holdout_censoring.csv')
    pred = pd.read_csv(HOLDOUT / 'holdout_censoring_pred.csv')
    assert set(metrics.model) == set(OFFLINE.values())
    assert (metrics.groupby(['model', 'set']).size() == 5).all()
    mc_picked = pred[(pred.set == 'acq') & pred.arms.fillna('').str.contains(r'\bmc')]
    assert len(mc_picked) == 1963 == metrics.loc[metrics.set == 'acq_mc', 'n_all'].iloc[0]
    assert (pred.set == 'random').sum() == 500
    return metrics, mc_picked


# Figures 1 and 2 were laid out as the two rows of one 4.05 in canvas. Their
# vertical positions are given as fractions of that canvas and rescaled by fy().
V4_HEIGHT = 4.05
DESIGN_HEIGHT = 1.62
TARGET_HEIGHT = 2.42
TARGET_OFFSET = .401  # bottom of the upper row


def design_figure():
    """Figure 1: the active learning protocol (a) and the ten strategies (b)."""
    def fy(y):
        return y * V4_HEIGHT / DESIGN_HEIGHT

    fig = plt.figure(figsize=(WIDTH, DESIGN_HEIGHT))
    fc.panel_box(fig, .006, fy(.006), .670, fy(.395))
    fc.panel_box(fig, .680, fy(.006), .994, fy(.395))
    ax = fig.add_axes([.045, fy(.032), .93, fy(.333)])
    ax.set(xlim=(-.8, 100), ylim=(0, 26.5))
    ax.axis('off')
    ax.text(-2.9, 26.4, 'a   Active learning protocol', fontsize=8.5, fontweight='bold',
            va='top')
    ax.text(70.1, 26.4, 'b   The ten strategies', fontsize=8.5, fontweight='bold',
            va='top')
    nodes = [('Evaluated data', '1,000 common\ninitial compositions'),
             ('Fit', 'WAE generator and\nsurrogate, per cycle'),
             ('Acquire', 'Thompson draw over\n25,000; select 10'),
             ('Evaluate', 'SQS, CHGNet, phonons\n' r'$\rightarrow$ AR and FR $G(v)$')]
    start, pitch, box_w = -1.5, 16.9, 15.6
    for j, (title, body) in enumerate(nodes):
        x = start + pitch * j
        ax.add_patch(FancyBboxPatch((x, 7.4), box_w, 10.6, boxstyle='round,pad=.3',
                                    ec='#7C919F', fc='#F2F6F8', lw=.7, clip_on=False))
        ax.text(x + box_w / 2, 15.6, title, ha='center', fontweight='bold', fontsize=7)
        ax.text(x + box_w / 2, 11.4, body, ha='center', va='center', fontsize=6.2,
                linespacing=1.35)
        if j < 3:
            ax.annotate('', xy=(x + pitch, 12.6), xytext=(x + box_w + .3, 12.6),
                        arrowprops=dict(arrowstyle='-|>', lw=.7, color='#555555',
                                        mutation_scale=6))
    head_x, tail_x = start + box_w / 2, start + 3 * pitch + box_w / 2
    ax.plot([tail_x, tail_x, head_x], [7.2, 3.4, 3.4], color='#555555', lw=.7)
    ax.annotate('', xy=(head_x, 7.2), xytext=(head_x, 3.4),
                arrowprops=dict(arrowstyle='-|>', lw=.7, color='#555555', mutation_scale=6))
    ax.text((head_x + tail_x) / 2, 3.4, '20 acquisition cycles', ha='center', va='center',
            fontsize=6.5, bbox=dict(fc='white', ec='none', pad=1.2))

    x0, dx = 75.5, 6.7
    for i, rep in enumerate(SHORT):
        ax.text(x0 + dx * i, 19.2, rep, ha='center', va='bottom', fontsize=6.0,
                linespacing=1.12)
    for j, (family, arms) in enumerate(FAMILIES.items()):
        y = 16.0 - 4.3 * j
        ax.text(x0 - 5.2, y, family, color=COLORS[family], fontweight='bold',
                va='center', ha='left', fontsize=7)
        for i in range(4):
            filled = i < len(arms)
            ax.scatter(x0 + dx * i, y, s=24, marker=MARKERS[i],
                       facecolors=COLORS[family] if filled else 'none',
                       edgecolors=COLORS[family] if filled else '#C4C4C4', lw=.7)
    ax.text(x0 - 5.2, .9, 'Filled: five runs, 20 cycles.\nOpen: not run.',
            fontsize=6.6, color=INK, linespacing=1.5)
    save(fig, 'fig01_design')


def target_figure(example):
    """Figure 2: one evaluated composition (a) and what each target predicts (b)."""
    def fy(y):
        return (y - TARGET_OFFSET) * V4_HEIGHT / TARGET_HEIGHT

    def fh(h):
        return h * V4_HEIGHT / TARGET_HEIGHT

    fig = plt.figure(figsize=(WIDTH, TARGET_HEIGHT))
    title_y = fy(.985)  # Shared heading height for the two panels.
    fc.panel_box(fig, .006, fy(.405), .377, fy(.998))
    fc.panel_box(fig, .387, fy(.405), .994, fy(.998))

    # --- a: one measured Gibbs landscape --------------------------------
    ax = fig.add_axes([.090, fy(.625), .225, fh(.275)])
    grid = np.linspace(0, 7, 200)
    ax.axvspan(7, 7.8, color='#EFEFEF', lw=0, zorder=0)
    ax.text(7.42, -3.0, 'outside window', ha='center', va='center', fontsize=6.2,
            color=MUTED, rotation=90)
    for values, coef, color, name, y_label, x_label in (
            (example['ar'], example['coef_ar'], '#B03A2E', 'AR', 4.7, 5.7),
            (example['fr'], example['coef_fr'], '#2E5F8A', 'FR', -5.7, 5.8)):
        ax.plot(grid, np.polyval(coef, grid), color=color, lw=1.1, zorder=2)
        ax.scatter(example['v'], values, s=10, color=color, zorder=3, lw=0)
        ax.text(x_label, y_label, name, color=color, fontsize=7.4, fontweight='bold',
                ha='center', va='center')
    base = -10.2
    for vstar, coef, color in ((example['v_ar'], example['coef_ar'], '#B03A2E'),
                               (example['v_fr'], example['coef_fr'], '#2E5F8A')):
        y = np.polyval(coef, vstar)
        ax.scatter(vstar, y, s=26, marker='v', color=color, zorder=4)
        ax.plot([vstar, vstar], [base, y], ls=':', lw=.7, color=color, zorder=1)
    ax.annotate('', xy=(example['v_fr'], base), xytext=(example['v_ar'], base),
                arrowprops=dict(arrowstyle='<|-|>', lw=.9, color=INK,
                                mutation_scale=6, shrinkA=0, shrinkB=0))
    ax.text((example['v_ar'] + example['v_fr']) / 2, base + .5,
            rf'$\Delta v={example["v_fr"] - example["v_ar"]:.2f}$', ha='center',
            va='bottom', fontsize=7)
    ax.text(example['v_ar'], base - .55, r'$v^{*}_{\mathrm{AR}}$', ha='center', va='top',
            fontsize=6.6, color='#B03A2E')
    ax.text(example['v_fr'], base - .55, r'$v^{*}_{\mathrm{FR}}$', ha='center', va='top',
            fontsize=6.6, color='#2E5F8A')
    kv_x = 3.0
    ax.annotate('', xy=(kv_x, np.polyval(example['coef_fr'], kv_x)),
                xytext=(kv_x, np.polyval(example['coef_ar'], kv_x)),
                arrowprops=dict(arrowstyle='<|-|>', lw=.8, color=MUTED, mutation_scale=5))
    ax.text(kv_x + .22, -3.1, r'$\approx\kappa v$', fontsize=6.4, color=MUTED)
    ax.set(xlim=(-.35, 7.8), ylim=(-12.9, 7.2), xticks=[0, 2, 4, 6, 7],
           yticks=[-10, -5, 0, 5], xlabel='Oxygen vacancies $v$',
           ylabel='$G$ relative to $v=0$ (eV)')
    ax.xaxis.set_label_coords(.5, -.17)
    ax.set_axisbelow(True)
    fig.text(.018, title_y, 'a   One evaluated composition', fontsize=8.5,
             fontweight='bold', va='top')

    fig.text(.018, fy(.512),
             r'OTC $=100\,\Delta v\,M_{\mathrm{O}}/M_{\mathrm{cell}}$ = '
             f'{example["otc"]:.2f} wt.%', fontsize=7.2)
    fig.text(.018, fy(.464),
             r'Ceiling $C=100\times 7\,M_{\mathrm{O}}/M_{\mathrm{cell}}$ = '
             f'{example["cap"]:.2f} wt.%', fontsize=7.2)
    fig.text(.018, fy(.416),
             'AR 1223 K, 0.2 atm    FR 1323 K, $10^{-14}$ atm',
             fontsize=6.6, color=INK)

    # --- b: what each representation predicts ----------------------------
    fig.text(.398, title_y, 'b   What the surrogate predicts', fontsize=8.5,
             fontweight='bold', va='top')
    left, w, gap = .398, .1255, .026
    bottom, height = fy(.645), fh(.215)
    boxes = [left + i * (w + gap) for i in range(4)]

    for k in (0, 1):
        ax = fc.mini_axes(fig, [boxes[k], bottom, w, height])
        ax.set(xlim=(0, 10), ylim=(0, 10))
        x = np.linspace(0, 10, 200)
        dens = np.exp(-((x - 6.2) / 1.9) ** 2)
        cap_x = 6.8
        if k == 0:
            ax.fill_between(x, 0, 7.2 * dens, color='#C7D3DB', lw=0)
            ax.fill_between(x[x >= cap_x], 0, 7.2 * dens[x >= cap_x], color='#B4485A',
                            alpha=.9, lw=0)
        else:
            keep = x < cap_x
            ax.fill_between(x[keep], 0, 7.2 * dens[keep], color='#C7D3DB', lw=0)
            ax.add_patch(Rectangle((cap_x - .3, 0), .6, 8.2, color='#536D7E', lw=0))
        ax.axvline(cap_x, color=INK, ls='--', lw=.8)
        ax.text(cap_x + .3, 9.2, '$C$', fontsize=7, color=INK, va='top')
        ax.set_xlabel('Sampled score', labelpad=1.5, fontsize=6.4)

    for k, (title, note) in enumerate((
            ('Scalar', 'Predicts OTC directly\nScore may exceed $C$'),
            ('Clipped scalar', 'Same model, score\nclipped to $C$*'),
            ('Gibbs vector', '8 AR and 8 FR levels\nOTC from the minima'),
            ('Vacancy input', r'$v$ as an input; FR from'
                              '\n' r'$G_{\mathrm{AR}}-\kappa v$'))):
        fig.text(boxes[k] + w / 2, bottom + height + fh(.040), title, ha='center',
                 fontsize=7.4, fontweight='bold')
        fig.text(boxes[k] + w / 2, bottom - fh(.048), note, ha='center', va='top',
                 fontsize=6.4, linespacing=1.4)

    fig.patches.append(FancyArrowPatch(
        (boxes[0] + w + .003, bottom + height * .55),
        (boxes[1] - .003, bottom + height * .55),
        transform=fig.transFigure, arrowstyle='-|>', mutation_scale=7, lw=.8, color=INK))
    fig.text((boxes[0] + w + boxes[1]) / 2, bottom + height * .55 + fh(.014), 'clip',
             ha='center', fontsize=6.4, color=INK)

    for k in (2, 3):
        ax = fc.mini_axes(fig, [boxes[k], bottom, w, height])
        vv = np.linspace(0, 7, 120)
        curvature = fc.KAPPA / 10
        ar_curve = curvature * (vv - 1.0) ** 2
        fr_curve = ar_curve - fc.KAPPA * vv
        ax.plot(vv, ar_curve, color='#B03A2E', lw=1.0)
        ax.plot(vv, fr_curve, color='#2E5F8A', lw=1.0,
                ls='-' if k == 2 else (0, (3, 1.6)))
        if k == 2:
            ax.scatter(np.arange(8), curvature * (np.arange(8) - 1.0) ** 2, s=6,
                       color='#B03A2E', lw=0, zorder=3)
            ax.scatter(np.arange(8), curvature * (np.arange(8) - 1.0) ** 2
                       - fc.KAPPA * np.arange(8), s=6,
                       color='#2E5F8A', lw=0, zorder=3)
        else:
            ax.annotate('', xy=(4.3, curvature * (4.3 - 1.0) ** 2 - fc.KAPPA * 4.3),
                        xytext=(4.3, curvature * (4.3 - 1.0) ** 2),
                        arrowprops=dict(arrowstyle='-|>', lw=.7, color=MUTED,
                                        mutation_scale=5))
            ax.text(4.55, -2.9, r'$-\kappa v$', fontsize=6.4, color=MUTED)
        ax.annotate('', xy=(6.0, -8.1), xytext=(1.0, -8.1),
                    arrowprops=dict(arrowstyle='<|-|>', lw=.8, color=INK, mutation_scale=5))
        ax.text(3.5, -7.8, r'$\Delta v$', ha='center', va='bottom', fontsize=6.5)
        ax.set(xlim=(-.4, 7.5), ylim=(-9.3, 7.9))
        ax.axvline(7, color='#999999', lw=.7, ls=':')
        ax.set_xlabel('$v$', labelpad=1.5, fontsize=6.4)

    bar_y = bottom - fh(.137)
    fig.patches.append(FancyBboxPatch(
        (boxes[2] - .006, bar_y - fh(.0135)), boxes[3] + w - boxes[2] + .012, fh(.029),
        boxstyle='round,pad=0.002', transform=fig.transFigure,
        facecolor='#EAF0F3', edgecolor='#9FB4BF', lw=.6))
    fig.text((boxes[2] + boxes[3] + w) / 2, bar_y,
             r'Score $\leq C$ by construction', ha='center', va='center',
             fontsize=6.9, color='#33505E')
    fig.text(.398, fy(.455), '*MC rounding changes the threshold; see Fig. 4.',
             fontsize=6.6, color=INK)
    save(fig, 'fig02_target')


def space_panels(fig, axes, flexible, margin_pt=3, gap_pt=6, pad_pt=5, inset_pt=6):
    """Place one row of axes so that their frames have equal gaps and page margins.

    Frames are measured as `fc.frame_panels` draws them: the axes' rendered content
    plus `pad_pt`, widened where the left-hand title, moved to the frame corner,
    is longer. Only the axes in `flexible` change width. The others keep it, since
    their row labels sit at fixed offsets. All axes get the same height.
    """
    canvas = fc.FigureCanvasAgg(fig)
    w, h = fig.bbox.width, fig.bbox.height
    mx, my = margin_pt * fig.dpi / 72 / w, margin_pt * fig.dpi / 72 / h
    gx, inset = gap_pt * fig.dpi / 72 / w, inset_pt * fig.dpi / 72 / w
    px, py = pad_pt * fig.dpi / 72 / w, pad_pt * fig.dpi / 72 / h
    to_fig = fig.transFigure.inverted()
    for _ in range(3):  # Tick labels shift slightly as the axes resize, so measure again.
        canvas.draw()
        renderer = canvas.get_renderer()
        frames, titles = [], []
        for ax in axes:
            title = ax._left_title
            titles.append(title.get_window_extent(renderer).transformed(to_fig))
            title.set_visible(False)
            content = ax.get_tightbbox(renderer).transformed(to_fig)
            title.set_visible(True)
            x0 = content.x0 - px
            frames.append((x0, max(content.x1 + px, x0 + inset + titles[-1].width + px),
                           content))
        boxes = [ax.get_position() for ax in axes]
        widths = [x1 - x0 for x0, x1, _ in frames]
        extra = 1 - 2 * mx - (len(axes) - 1) * gx - sum(widths)
        stretch = sum(boxes[k].width for k, ax in enumerate(axes) if ax in flexible)
        bottom = max(b.y0 - c.y0 for b, (_, _, c) in zip(boxes, frames))
        top = max(max(c.y1, t.y1) - b.y1 for b, (_, _, c), t in zip(boxes, frames, titles))
        y0 = my + py + bottom
        height = 1 - my - py - top - y0
        x = mx
        for ax, b, (x0, x1, _) in zip(axes, boxes, frames):
            grow = b.width * extra / stretch if ax in flexible else 0
            ax.set_position([x + b.x0 - x0, y0, b.width + grow, height])
            x += x1 - x0 + grow + gx


def surrogate_value_figure(runs, ceil_runs, random_ref, metrics, mc_picked):
    fig = plt.figure(figsize=(WIDTH, 2.95))
    rep_labels = [REPS[META[a][1]] for a in OFFLINE_ARMS]

    # a: realized OTC against the ceiling, per run
    ax = fig.add_axes([.075, .15, .27, .70])
    lims = (3.0, 7.0)
    ax.plot(lims, lims, color=MUTED, ls='--', lw=.7, zorder=1)
    ax.text(5.55, 5.92, 'OTC = C', color=MUTED, fontsize=6.4, ha='right', va='top')
    for arm in ARMS:
        family, i = META[arm]
        r = runs[runs.arm == arm]
        ax.scatter(r.mean_cap, r.mean_otc, color=COLORS[family], marker=MARKERS[i],
                   s=14, lw=0, alpha=.9, zorder=3)
    ax.scatter(ceil_runs.mean_cap, ceil_runs.mean_otc, color=CEIL_COLOR, marker='X',
               s=26, lw=0, zorder=4)
    random_cap = 100 * 7 * fc.O_MASS / random_ref.mass
    ax.scatter(random_cap.mean(), random_ref.otc.mean(), facecolors='white',
               edgecolors=RANDOM_COLOR, s=24, lw=1, zorder=4)
    ax.annotate('Random reference', (random_cap.mean(), random_ref.otc.mean()),
                xytext=(8, -1), textcoords='offset points', fontsize=6.4, color=MUTED,
                va='center')
    ax.annotate('Ceiling only', (ceil_runs.mean_cap.mean(), ceil_runs.mean_otc.max()),
                xytext=(-4, 9), textcoords='offset points', fontsize=6.4, color=CEIL_COLOR,
                ha='right', va='bottom')
    ax.set(xlim=lims, ylim=(0, 6.0), xticks=[3, 4, 5, 6, 7], yticks=[0, 2, 4, 6],
           xlabel='Mean ceiling C (wt.%)', ylabel='Mean acquired OTC (wt.%)')
    panel(ax, 'a', 'OTC and ceiling, per run', grid='both')
    axes = [ax]

    # b: offline ranking quality on two test sets
    ax = fig.add_axes([.535, .15, .19, .70])
    spearman = metrics.groupby(['model', 'set']).spearman_all.agg(['mean', 'min', 'max'])
    for j, arm in enumerate(OFFLINE_ARMS):
        family, i = META[arm]
        model = OFFLINE[arm]
        r, a = spearman.loc[(model, 'random')], spearman.loc[(model, 'acq_mc')]
        ax.plot([a['mean'], r['mean']], [j, j], color=COLORS[family], lw=1.2, zorder=2)
        ax.scatter(r['mean'], j, facecolors='white', edgecolors=COLORS[family],
                   marker=MARKERS[i], s=18, lw=.9, zorder=3)
        ax.scatter(a['mean'], j, color=COLORS[family], marker=MARKERS[i], s=18, lw=0,
                   zorder=3)
    ax.set(xlim=(.3, 1.0), xticks=[.4, .6, .8, 1.0], xlabel='Spearman, predicted vs evaluated')
    row_axis(ax, rep_labels, None, None, gap_after=(3,))
    ax.set_ylim(ROW_LIM)
    panel(ax, 'b', 'Ranking held-out compositions', grid='x')
    for family in ('DE', 'GP'):
        family_brace(ax, None, [OFFLINE_ARMS.index(a) for a in FAMILIES[family]], family)
    ax.legend(handles=[
        Line2D([], [], ls='', marker='o', mfc='white', mec=INK, ms=4, label='Random'),
        Line2D([], [], ls='', marker='o', color=INK, ms=4, label='MC-acquired')],
        loc='upper center', bbox_to_anchor=(.5, 1.0), ncol=2, frameon=False, fontsize=6.2,
        handletextpad=.1, columnspacing=.8, borderaxespad=0)
    axes.append(ax)

    # c: predicted / ceiling where the true OTC is at the ceiling
    ax = fig.add_axes([.775, .15, .205, .70])
    true = mc_picked.otc.to_numpy()
    ceil = mc_picked.ceil.to_numpy()
    at_ceiling = np.isclose(true, ceil, rtol=1e-3)
    for j, arm in enumerate(OFFLINE_ARMS):
        family, i = META[arm]
        model = OFFLINE[arm]
        ratio = np.concatenate([mc_picked[f'{model}_s{k}'].to_numpy()[at_ceiling]
                                / ceil[at_ceiling] for k in range(1, 6)])
        q05, q25, q50, q75, q95 = np.quantile(ratio, [.05, .25, .5, .75, .95])
        color = COLORS[family]
        ax.plot([q05, q95], [j, j], color=color, lw=.8, zorder=2)
        ax.add_patch(Rectangle((q25, j - .22), q75 - q25, .44,
                                                  facecolor=color, alpha=.35, lw=0, zorder=2))
        ax.scatter(q50, j, color=color, marker=MARKERS[i], s=18, lw=0, zorder=3)
    ax.axvline(1, color=INK, lw=.8, ls='--', zorder=1)
    ax.set(xlim=(.55, 1.3), xticks=[.6, .8, 1.0, 1.2], xlabel='Predicted OTC / ceiling C')
    row_axis(ax, [''] * len(OFFLINE_ARMS), None, None, gap_after=(3,))
    ax.set_ylim(ROW_LIM)
    panel(ax, 'c', 'At the ceiling', grid='x')
    axes.append(ax)

    space_panels(fig, axes, flexible=(axes[0], axes[2]))
    frame_panels(fig, [[[a] for a in axes]])
    save(fig, 'fig05_surrogate_value')


def individual_runs_figure(cycles, ceil_cycles, initial):
    fig, axes = plt.subplots(3, 4, figsize=(WIDTH, 5.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.985, top=.95, bottom=.08, hspace=.42, wspace=.14)
    initial_max = initial.otc.max()
    rows = list(FAMILIES.items()) + []
    panels = [(arm, f'{fam} · {REPS[META[arm][1]]}', COLORS[fam], cycles[cycles.arm == arm])
              for fam, arms in rows for arm in arms]
    panels.append(('ceil', 'Ceiling only', CEIL_COLOR, ceil_cycles))
    slots = [(0, c) for c in range(4)] + [(1, c) for c in range(4)] + [(2, 0), (2, 1), (2, 2)]
    for k, ((arm, title, color, table), (r, c)) in enumerate(zip(panels, slots)):
        ax = axes[r, c]
        for _, g in table.groupby('replicate'):
            y = np.maximum.accumulate(np.r_[initial_max, g.sort_values('cycle').cumulative_max])
            ax.plot(np.arange(21), y, color=color, lw=.8, alpha=.75)
        ax.axhline(initial_max, color='#999999', ls='--', lw=.6)
        ax.set(xlim=(0, 20), ylim=(4.7, 6.5), xticks=[0, 10, 20], yticks=[5, 5.5, 6, 6.5])
        panel(ax, chr(97 + k), title)
        if c == 0:
            ax.set_ylabel('Best OTC so far (wt.%)')
        if r == 2 or (r == 1 and c == 3):
            ax.set_xlabel('Acquisition cycle')
            ax.tick_params(labelbottom=True)
    axes[1, 3].set_xlabel('Acquisition cycle')
    axes[2, 3].axis('off')
    axes[2, 3].text(0, .95, 'Five runs per strategy.\nGray dashed line: best OTC\nin the initial dataset.\n\n'
                    'Ceiling only: no surrogate,\neach cycle selects the ten\ncandidates with the highest C.',
                    va='top', fontsize=6.6, color=INK, linespacing=1.5,
                    transform=axes[2, 3].transAxes)
    save(fig, 'figS01_individual_runs')


def endpoints_figure(runs, ceil_runs):
    labels = [REPS[META[a][1]] for a in ARMS] + ['Ceiling only']
    n = len(labels)
    fig = plt.figure(figsize=(WIDTH, 4.0))
    gs = fig.add_gridspec(1, 3, left=.15, right=.985, top=.92, bottom=.12, wspace=.10)
    spec = [('mean_otc', 'Mean acquired', (0, 3.0), (4.2, 5.45), [0, 1, 2, 3], [4.5, 5.0]),
            ('max_otc', 'Best acquired', (1.5, 4.0), (4.6, 6.5), [2, 3, 4], [5.0, 5.5, 6.0, 6.5]),
            ('top10_mean', 'Mean of top 10', (1.0, 4.5), (4.6, 6.3), [1, 2, 3, 4], [5.0, 5.5, 6.0])]
    for k, (col, title, left_lim, main_lim, left_ticks, main_ticks) in enumerate(spec):
        sub = gs[0, k].subgridspec(1, 2, width_ratios=[1, 2.6], wspace=.06)
        left, main = fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1], sharey=None)
        for ax in (left, main):
            for j, arm in enumerate(ARMS + ['ceil']):
                if arm == 'ceil':
                    v, color, marker = ceil_runs[col].sort_values().to_numpy(), CEIL_COLOR, 'X'
                else:
                    f, i = META[arm]
                    v = runs[runs.arm == arm][col].sort_values().to_numpy()
                    color, marker = COLORS[f], MARKERS[i]
                ax.scatter(v, j + np.linspace(-.27, .27, 5), color=color, marker=marker,
                           s=12, lw=0, zorder=3, clip_on=True)
                ax.plot([v.mean()] * 2, [j - .32, j + .32], color=INK, lw=.9, zorder=2)
            row_axis(ax, labels if (ax is left and k == 0) else [''] * n, None, None,
                     gap_after=(3, 7, 9))
            ax.grid(axis='x')
            ax.set_axisbelow(True)
        left.set(xlim=left_lim, xticks=left_ticks)
        main.set(xlim=main_lim, xticks=main_ticks)
        main.set_xlabel('OTC (wt.%)', x=.32)
        # Break marks between the two x ranges.
        d = .012
        for ax, x in ((left, 1), (main, 0)):
            ax.plot([x - d * (3.6 if x == 0 else 1), x + d * (3.6 if x == 0 else 1)], [-.02, .02],
                    transform=ax.transAxes, color='#666666', lw=.6, clip_on=False)
        left.set_title(f'{chr(97 + k)}   {title}', loc='left', pad=5)
        if k == 0:
            for family, arms in FAMILIES.items():
                family_brace(left, None, [ARMS.index(a) for a in arms], family)
    save(fig, 'figS02_run_endpoints')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fc.style()
    acq, initial, runs, cycles, scores = fc.load()
    example = fc.example_landscape()
    design_figure()
    target_figure(example)
    fc.discovery_figure(runs, cycles, acq, initial)
    fc.mechanism_figure(runs, acq, initial, scores)

    ceil_runs = pd.read_csv(DATA / 'ceiling_run_metrics.csv')
    ceil_cycles = pd.read_csv(DATA / 'ceiling_cycle_metrics.csv')
    random_ref = pd.read_csv(DATA / 'random_baseline_additional.csv')
    assert len(ceil_runs) == 5 and len(ceil_cycles) == 100 and len(random_ref) == 500
    metrics, mc_picked = load_offline()

    surrogate_value_figure(runs, ceil_runs, random_ref, metrics, mc_picked)
    individual_runs_figure(cycles, ceil_cycles, initial)
    endpoints_figure(runs, ceil_runs)
    plt.close('all')


if __name__ == '__main__':
    main()
