#!/usr/bin/env python3
"""Shared style, data loading and panel helpers of the figure scripts, and Figures 3 and 4.

Used by make_figures.py. Figures are drawn at their printed width of 180 mm with a
7 pt base font and saved as PDF in figures/.
"""
import json
import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox

from paths import DATA, FIGURES as OUT, HEOC

WIDTH = 180 / 25.4

O_MASS = 15.999
CEILING_VACANCIES = 7
KAPPA = 1.8836231615976065

FAMILIES = {'DE': ['de', 'de_clip', 'de_land', 'dde'],
            'GP': ['gp', 'gp_clip', 'mogp', 'dgp'],
            'MC': ['mc', 'mc_clip']}
FAMILY_NAMES = {'DE': 'Deep ensemble', 'GP': 'Gaussian process', 'MC': 'MC dropout'}
# Okabe-Ito hues chosen so the three families also separate in grayscale
# (8-bit luminance 119 / 87 / 151).
COLORS = {'DE': '#D55E00', 'GP': '#0072B2', 'MC': '#CC79A7'}
REPS = ['Scalar', 'Clipped scalar', 'Gibbs vector', 'Vacancy input']
SHORT = ['Scalar', 'Clipped\nscalar', 'Gibbs\nvector', 'Vacancy\ninput']
STYLES = ['-', '--', '-.', ':']
MARKERS = ['o', 's', '^', 'D']
ARMS = sum(FAMILIES.values(), [])
META = {arm: (fam, i) for fam, arms in FAMILIES.items() for i, arm in enumerate(arms)}

INK = '#222222'
MUTED = '#666666'
RULE = '#DDDDDD'
INITIAL_COLOR = '#8C8C8C'

# Figure 2a: one evaluated composition, named in the caption. Chosen because both
# quadratic minima lie strictly inside the v = 0..7 window, so the window edge is
# visible without binding, and its measured AR-FR offset matches kappa to 0.003 eV.
EXAMPLE = dict(db='spinel_rev_mogp_s5', tag_prefix='Al14CuLi11Mn10O41',
               formula='Al14CuLi11Mn10O48', arm='mogp', replicate=5, cycle=6)



# --------------------------------------------------------------------------
# infrastructure


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 7,
        'axes.labelsize': 7.2, 'axes.titlesize': 8.5, 'axes.titleweight': 'bold',
        'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8, 'legend.fontsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .6, 'axes.edgecolor': '#666666',
        'xtick.major.size': 2.6, 'ytick.major.size': 2.6,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'grid.color': '#E8E8E8', 'grid.linewidth': .45,
        'lines.linewidth': 1.3, 'patch.linewidth': .6,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.fontset': 'dejavusans',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white'})


def gray(hexcolor):
    r, g, b = (int(hexcolor[i:i + 2], 16) for i in (1, 3, 5))
    return 0.299 * r + 0.587 * g + 0.114 * b


def panel(ax, letter, title='', grid='y'):
    ax.set_title(f'{letter}   {title}'.rstrip(), loc='left', pad=5)
    if grid:
        ax.grid(axis=grid)
    ax.set_axisbelow(True)


def row_axis(ax, labels, group_sizes, colors, gap_after=(), label_pad=True):
    """Strategy rows with representation-only labels and a family rule."""
    ax.set_ylim(len(labels) - .5, -.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels if label_pad else [])
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    for y in gap_after:
        ax.axhline(y + .5, color=RULE, lw=.6, zorder=0)
    return ax


def family_brace(ax, x, rows, family, fontsize=7):
    """Vertical rule plus rotated family name just outside the row labels."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    labels = [t.get_window_extent(renderer) for t in ax.get_yticklabels() if t.get_text()]
    # Locate the rule from rendered labels, rather than placing it through them.
    left = min(b.x0 for b in labels) - 5 * ax.figure.dpi / 72
    x = ax.transAxes.inverted().transform((left, 0))[0]
    text_gap = (5 * ax.figure.dpi / 72) / ax.bbox.width
    top, bottom = min(rows) - .34, max(rows) + .34
    ax.plot([x, x], [top, bottom], color=COLORS[family], lw=1.4,
            solid_capstyle='butt', clip_on=False, transform=ax.get_yaxis_transform())
    ax.text(x - text_gap, (top + bottom) / 2, family, rotation=90, va='center', ha='center',
            color=COLORS[family], fontsize=fontsize, fontweight='bold',
            clip_on=False, transform=ax.get_yaxis_transform())


def save(fig, name):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    bounds = fig.get_tightbbox(canvas.get_renderer())
    w, h = fig.get_size_inches()
    fits = bounds.x0 >= -1e-6 and bounds.y0 >= -1e-6 and bounds.x1 <= w + 1e-6 and bounds.y1 <= h + 1e-6
    assert fits, f'{name}: content extends outside the fixed canvas'
    fig.savefig(OUT / f'{name}.pdf', dpi=300)
    print(f'{name}  {w * 25.4:.0f} x {h * 25.4:.0f} mm  fits={fits}', flush=True)


# --------------------------------------------------------------------------
# data


def load():
    acq = pd.read_csv(DATA / 'acquisitions.csv')
    initial = pd.read_csv(DATA / 'initial_dataset.csv')
    runs = pd.read_csv(DATA / 'run_metrics.csv')
    cycles = pd.read_csv(DATA / 'cycle_metrics.csv')
    scores = pd.read_csv(DATA / 'selected_scores_all.csv')

    counts = acq.groupby(['arm', 'replicate']).size()
    assert len(acq) == 9982 and len(initial) == 1000 and len(runs) == 50
    assert all(counts[arm].sum() == 1000 for arm in ARMS if not arm.startswith('mc'))
    assert counts['mc'].sum() == 995 and counts['mc_clip'].sum() == 987
    cations = [c for c in acq.columns if c.startswith('count_')]
    assert (acq[cations].sum(axis=1) == 36).all()
    assert (initial[cations].sum(axis=1) == 36).all()
    assert acq.count_Li.max() == 12 and initial.count_Li.max() == 12
    for frame in (acq, initial):
        assert np.allclose(frame.otc, 100 * frame.vacancy_diff * O_MASS / frame.mass, atol=1e-9)
        assert np.allclose(frame.cap, 100 * CEILING_VACANCIES * O_MASS / frame.mass, atol=1e-9)
    assert set(scores.arm) == set(ARMS) and len(scores) == 10000
    assert (scores.groupby(['arm', 'replicate']).size() == 200).all()
    assert not (scores.above_cap & scores.at_cap).any()
    # The audit flags use a tolerance of 1e-5 OTC percentage points.
    assert np.array_equal(scores.above_cap, scores.score > scores.cap + 1e-5)
    assert np.array_equal(scores.at_cap, np.abs(scores.score - scores.cap) <= 1e-5)
    score_status = scores.groupby('arm')[['above_cap', 'at_cap']].mean()
    assert np.isclose(score_status.loc['mc', 'above_cap'], .862)
    assert np.isclose(score_status.loc['mc_clip', 'above_cap'], .426)
    assert (score_status.loc[['de_clip', 'gp_clip', 'de_land', 'dde', 'mogp', 'dgp'],
                             'above_cap'] == 0).all()

    grays = sorted(gray(c) for c in COLORS.values())
    assert min(np.diff(grays)) >= 30, 'family colors must separate in grayscale'
    return acq, initial, runs, cycles, scores


def example_landscape():
    """Read one measured Gibbs landscape and reproduce the pipeline's extraction."""
    path = HEOC / EXAMPLE['db'] / 'training.db'
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        rows = [json.loads(kv) for (kv,) in con.execute('select key_value_pairs from systems')]
    finally:
        con.close()
    kv = next(r for r in rows if r.get('tag', '').startswith(EXAMPLE['tag_prefix'])
              and r.get('cycle') == EXAMPLE['cycle'])
    v = np.arange(8)
    ar = np.array([kv[f'gibbs_AR_{i}'] for i in v])
    fr = np.array([kv[f'gibbs_FR_{i}'] for i in v])
    coef_ar, coef_fr = np.polyfit(v, ar, 2), np.polyfit(v, fr, 2)

    def minimum(coef):
        a, b = coef[0], coef[1]
        if abs(a) < 1e-10:
            return 0.0 if b > 0 else 7.0
        if a < 0:
            return 0.0 if np.polyval(coef, 0) < np.polyval(coef, 7) else 7.0
        return float(np.clip(-b / (2 * a), 0, 7))

    v_ar, v_fr = minimum(coef_ar), minimum(coef_fr)
    otc = kv['otc_quadratic_fit'] * 100
    assert abs((v_fr - v_ar) - kv['vacancy_diff_quadratic_fit']) < 1e-9
    offset = np.polyfit(v, ar - fr, 1)
    return dict(v=v, ar=ar, fr=fr, coef_ar=coef_ar, coef_fr=coef_fr, v_ar=v_ar, v_fr=v_fr,
                otc=otc, cap=otc * CEILING_VACANCIES / (v_fr - v_ar),
                offset_slope=float(offset[0]),
                offset_residual=float(np.abs((ar - fr) - np.polyval(offset, v)).max()))


# --------------------------------------------------------------------------
# panel helpers


def mini_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.set_facecolor('#FAFAFA')
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#B0B0B0')
    return ax


def panel_box(fig, x0, y0, x1, y1):
    """Light gray frame that groups one panel's elements, in figure coordinates."""
    fig.patches.append(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0,rounding_size=.008',
        transform=fig.transFigure, facecolor='none', edgecolor='#BDBDBD', lw=.7,
        zorder=-10))


def frame_panels(fig, rows, columns=(), lefts=(), rights=(), pad_pt=5, inset_pt=6):
    """Frame each panel with a box sized to its rendered content.

    `rows` lists the panels row by row, each panel a list of axes. Frames in one row
    share their top and bottom, and panels listed together in `columns` (by their
    first axes) share their left and right edges, or only one of them for panels
    listed in `lefts` or `rights`, as when a full-width panel sits under a row. Each left-hand title is moved to
    the frame's top-left corner so that all three figures place panel letters alike.
    """
    canvas = FigureCanvasAgg(fig)
    w, h = fig.bbox.width, fig.bbox.height
    pad_x, pad_y = pad_pt * fig.dpi / 72 / w, pad_pt * fig.dpi / 72 / h
    for _ in range(2):  # Second pass measures again after the titles have moved.
        canvas.draw()
        renderer = canvas.get_renderer()
        frames = []
        for row in rows:
            extents = [Bbox.union([ax.get_tightbbox(renderer) for ax in group])
                       .transformed(fig.transFigure.inverted()) for group in row]
            y0, y1 = min(b.y0 for b in extents), max(b.y1 for b in extents)
            frames.append([[b.x0 - pad_x, y0 - pad_y, b.x1 + pad_x, y1 + pad_y]
                           for b in extents])
        where = {group[0]: frame for row, row_frames in zip(rows, frames)
                 for group, frame in zip(row, row_frames)}
        for sides, groups in (((0, 2), columns), ((0,), lefts), ((2,), rights)):
            for group in groups:
                for side in sides:
                    edge = (min if side == 0 else max)(where[ax][side] for ax in group)
                    for ax in group:
                        where[ax][side] = edge
        for row, row_frames in zip(rows, frames):
            for group, (x0, _, _, _) in zip(row, row_frames):
                ax = group[0]
                title = ax.get_title(loc='left')
                if title:
                    left = x0 + inset_pt * fig.dpi / 72 / w
                    x = ax.transAxes.inverted().transform(fig.transFigure.transform((left, 0)))[0]
                    ax.set_title(title, loc='left', pad=5, x=x)
    for row_frames in frames:
        for frame in row_frames:
            panel_box(fig, *frame)


# --------------------------------------------------------------------------
# Figure 3: discovery at a fixed budget


def discovery_figure(runs, cycles, acq, initial):
    fig = plt.figure(figsize=(WIDTH, 5.75))
    gs = fig.add_gridspec(3, 3, left=.1, right=.985, bottom=.085, top=.955,
                          height_ratios=[1, 1, .85], width_ratios=[1, 1, .62],
                          hspace=.66, wspace=.2)
    initial_max = initial.otc.max()
    thresholds = np.linspace(5, 6.5, 301)
    tail_axes, top_axes = {}, []

    for k, (family, arms) in enumerate(FAMILIES.items()):
        top = fig.add_subplot(gs[0, k])
        tail = fig.add_subplot(gs[1, k]) if family != 'MC' else None
        tail_axes[family] = tail
        top_axes.append(top)
        color = COLORS[family]
        for arm in arms:
            _, i = META[arm]
            table = cycles[cycles.arm == arm].pivot(index='cycle', columns='replicate',
                                                    values='cumulative_max_with_initial')
            values = np.vstack([np.full((1, 5), initial_max), table.to_numpy()])
            x = np.arange(21)
            top.fill_between(x, values.min(1), values.max(1), color=color, alpha=.11, lw=0)
            top.plot(x, values.mean(1), color=color, ls=STYLES[i], marker=MARKERS[i],
                     ms=2.8, markevery=[4, 8, 12, 16, 20], lw=1.35)
            per_run = []
            for replicate, g in acq[acq.arm == arm].groupby('replicate'):
                frac = (g.otc.to_numpy()[:, None] >= thresholds).mean(0) * 100
                per_run.append(frac)
                if tail is not None:
                    tail.plot(thresholds, np.where(frac > 0, frac, np.nan), color=color,
                              ls=STYLES[i], lw=.55, alpha=.38)
            mean = np.mean(per_run, 0)
            if tail is not None:
                tail.plot(thresholds, np.where(mean > 0, mean, np.nan), color=color,
                          ls=STYLES[i], lw=1.5, marker=MARKERS[i], markevery=70, ms=2.8)
        top.axhline(initial_max, color='#999999', ls='--', lw=.7)
        top.set(xlim=(-.3, 20.6), ylim=(4.68, 6.52), xticks=[0, 5, 10, 15, 20],
                yticks=[5, 5.5, 6, 6.5], xlabel='Acquisition cycle')
        panel(top, chr(97 + k), FAMILY_NAMES[family])
        if k == 0:
            top.set_ylabel('Best OTC so far (wt.%)')
        else:
            top.tick_params(labelleft=False)
        if tail is not None:
            tail.set(yscale='log', xlim=(4.97, 6.55), ylim=(.08, 160),
                     xticks=[5, 5.5, 6, 6.5], yticks=[.1, 1, 10, 100],
                     xlabel='OTC threshold (wt.%)')
            tail.set_yticklabels(['0.1', '1', '10', '100'])
            tail.minorticks_off()
            panel(tail, chr(100 + k), f'{family} upper tail')
            if k == 0:
                tail.set_ylabel('Acquisitions $\\geq$\nthreshold (%)')
            else:
                tail.tick_params(labelleft=False)

    note = fig.add_subplot(gs[1, 2])
    note.axis('off')
    handles = [Line2D([], [], color='#333333', ls=STYLES[i], marker=MARKERS[i], ms=3,
                      lw=1.2, label=rep) for i, rep in enumerate(REPS)]
    note.legend(handles=handles, loc='upper left', bbox_to_anchor=(-.02, 1.02),
                frameon=False, handlelength=2.9, labelspacing=.65, borderpad=0)
    note.text(-.02, .30, 'Thin lines in d and e:\nindividual runs.\n\n'
                         'Neither MC strategy\nreached 6.0 wt.%.',
              transform=note.transAxes, va='top', fontsize=6.6, linespacing=1.5,
              color=INK)

    ax = fig.add_subplot(gs[2, :])
    ticks, labels, spans, pos = [], [], [], 0
    for family, arms in FAMILIES.items():
        spans.append((pos, pos + len(arms) - 1, family))
        for arm in arms:
            _, i = META[arm]
            values = runs[runs.arm == arm].sort_values('mean_otc').mean_otc.to_numpy()
            ax.scatter(pos + np.linspace(-.17, .17, 5), values, s=15, marker=MARKERS[i],
                       color=COLORS[family], edgecolors='white', lw=.3, zorder=3)
            ax.plot([pos - .28, pos + .28], [values.mean()] * 2, color=INK, lw=1.2, zorder=2)
            ticks.append(pos)
            labels.append(SHORT[i])
            pos += 1
        pos += .7
    ax.set(xticks=ticks, xticklabels=labels, xlim=(-.6, pos - 1.1),
           ylim=(4.15, 5.52), yticks=[4.2, 4.6, 5.0, 5.4],
           ylabel='Mean acquired\nOTC (wt.%)')
    for lo, hi, family in spans:
        ax.text((lo + hi) / 2, 5.47, family, ha='center', va='top', color=COLORS[family],
                fontweight='bold', fontsize=7.4)
    for lo, _, _ in spans[1:]:
        ax.axvline(lo - .85, color=RULE, lw=.6)
    panel(ax, 'f', 'Acquisition quality over cycles 1 to 20')
    frame_panels(fig, [[[a] for a in top_axes],
                       [[tail_axes['DE']], [tail_axes['GP']]], [[ax]]],
                 columns=[[top_axes[0], tail_axes['DE']], [top_axes[1], tail_axes['GP']]],
                 lefts=[[top_axes[0], tail_axes['DE'], ax]], rights=[[top_axes[2], ax]])

    save(fig, 'fig03_discovery')


# --------------------------------------------------------------------------
# Figure 4: selected scores, ceilings and chemistry


def mechanism_figure(runs, acq, initial, scores):
    fig = plt.figure(figsize=(WIDTH, 6.2))
    gs = fig.add_gridspec(2, 3, left=.145, right=.975, bottom=.07, top=.95,
                          height_ratios=[1, 1.15], width_ratios=[1.2, .92, .86],
                          hspace=.34, wspace=.22)
    rep_labels = [REPS[META[a][1]] for a in ARMS]
    status_colors = ['#C9D6DE', '#536D7E', '#B4485A']

    # a: selected acquisition scores against the ceiling
    ax = fig.add_subplot(gs[0, 0])
    for j, arm in enumerate(ARMS):
        s = scores[scores.arm == arm]
        fractions = [float((~(s.above_cap | s.at_cap)).mean()), float(s.at_cap.mean()),
                     float(s.above_cap.mean())]
        assert np.isclose(sum(fractions), 1)
        left = 0.
        for f, color in zip(fractions, status_colors):
            ax.barh(j, f * 100, left=left * 100, height=.56, color=color, lw=0)
            left += f
        if fractions[2] > .10:
            ax.text(100 - fractions[2] * 50, j, f'{fractions[2] * 100:.1f}%', ha='center',
                    va='center', fontsize=6.2, color='white', fontweight='bold')
        if fractions[1] > .15:
            ax.text(100 * (fractions[0] + fractions[1] / 2), j,
                    f'{fractions[1] * 100:.1f}%', ha='center', va='center', fontsize=6.2,
                    color='white', fontweight='bold')
    ax.set(xlim=(0, 100), xticks=[0, 50, 100], xlabel='Selected scores (%)')
    row_axis(ax, rep_labels, None, None, gap_after=(3, 7))
    panel(ax, 'a', 'Score against ceiling', grid='x')
    panel_axes = [ax]
    for family, arms in FAMILIES.items():
        rows = [ARMS.index(a) for a in arms]
        family_brace(ax, -.40, rows, family)

    # b: evaluated OTC and its ceiling, per run
    ax = fig.add_subplot(gs[0, 1])
    for j, arm in enumerate(ARMS):
        family, i = META[arm]
        color = COLORS[family]
        table = runs[runs.arm == arm].sort_values('replicate')
        for offset, (_, r) in zip(np.linspace(-.27, .27, 5), table.iterrows()):
            ax.plot([r.mean_otc, r.mean_cap], [j + offset] * 2, color=color, alpha=.42, lw=.6)
        mean_otc, mean_cap = table.mean_otc.mean(), table.mean_cap.mean()
        ax.plot([mean_otc, mean_cap], [j, j], color=color, lw=1.4, zorder=3)
        ax.scatter(mean_otc, j, s=19, color=color, zorder=4, lw=0)
        ax.scatter(mean_cap, j, s=19, facecolors='white', edgecolors=color, lw=.9, zorder=4)
    ax.set(xlim=(4.18, 5.92), xticks=[4.5, 5.0, 5.5], xlabel='OTC (wt.%)')
    row_axis(ax, [''] * len(ARMS), None, None, gap_after=(3, 7))
    panel(ax, 'b', 'OTC and ceiling', grid='x')
    panel_axes.append(ax)

    # c: window-edge saturation, per run
    ax = fig.add_subplot(gs[0, 2])
    for j, arm in enumerate(ARMS):
        family, i = META[arm]
        table = runs[runs.arm == arm].sort_values('saturated_fraction')
        ax.scatter(table.saturated_fraction * 100, j + np.linspace(-.26, .26, 5),
                   color=COLORS[family], marker=MARKERS[i], s=13, lw=0, zorder=3)
        ax.plot([table.saturated_fraction.mean() * 100] * 2, [j - .31, j + .31],
                color=INK, lw=1, zorder=2)
    ax.set(xlim=(38, 92), xticks=[40, 60, 80], xlabel=r'Acquisitions at $\Delta v=7$ (%)')
    row_axis(ax, [''] * len(ARMS), None, None, gap_after=(3, 7))
    panel(ax, 'c', 'Window edge', grid='x')
    panel_axes.append(ax)

    # d, e: composition distributions, initial set as a reference row
    rows = ['initial'] + ARMS
    row_labels = ['Initial set'] + rep_labels
    for k, (key, grid, title, xlabel, xticks) in enumerate((
            ('count_Li', np.arange(13), 'Lithium content', 'Li atoms per 36 cations',
             [0, 4, 8, 12]),
            ('n_cations', np.arange(2, 9), 'Cation diversity', 'Distinct cation elements',
             [2, 4, 6, 8]))):
        ax = fig.add_subplot(gs[1, k])
        curves = []
        for arm in rows:
            groups = ([initial] if arm == 'initial'
                      else [g for _, g in acq[acq.arm == arm].groupby('replicate')])
            curves.append(np.asarray([[float((g[key] == x).mean()) for x in grid]
                                      for g in groups]))
        scale = .82 / max(c.max() for c in curves)
        for j, (arm, values) in enumerate(zip(rows, curves)):
            color = INITIAL_COLOR if arm == 'initial' else COLORS[META[arm][0]]
            mean = values.mean(0)
            ax.bar(grid, scale * mean, bottom=j - scale * mean, width=.8, color=color,
                   alpha=.8, lw=0)
            if arm != 'initial':
                ax.vlines(grid, j - scale * values.max(0), j - scale * values.min(0),
                          color='#333333', lw=.55)
        bar_x = grid[-1] + .75
        ax.plot([bar_x, bar_x], [-.05, -.05 - .5 * scale], color='#555555', lw=.8)
        ax.text(bar_x + .2, -.05 - .25 * scale, '50%', va='center', ha='left',
                fontsize=6.2, color=MUTED)
        ax.set_ylim(len(rows) - .35, -1.05)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(row_labels if k == 0 else [])
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        for y in (.5, 4.5, 8.5):
            ax.axhline(y, color=RULE, lw=.6, zorder=0)
        ax.set(xlim=(grid[0] - .8, grid[-1] + 1.3), xticks=xticks, xlabel=xlabel)
        panel(ax, chr(100 + k), title, grid=None)
        panel_axes.append(ax)
        if k == 0:
            for family, arms in FAMILIES.items():
                family_brace(ax, -.40, [rows.index(a) for a in arms], family)
            ax.axvline(12, color=INK, ls=(0, (4, 2)), lw=.8, zorder=4)
            ax.text(11.7, -.95, 'Limit: Li on all 12 A sites', ha='right', va='top',
                    fontsize=6.3, color=INK)

    legend_ax = fig.add_subplot(gs[1, 2])
    legend_ax.axis('off')
    score_legend = legend_ax.legend(handles=[Line2D([], [], color=c, lw=5, label=l) for c, l in
                              zip(status_colors, ['Below ceiling', 'At ceiling',
                                                  'Above ceiling'])],
                     loc='upper left', bbox_to_anchor=(0, 1.0), frameon=False,
                     handlelength=1.1, title='Panel a', title_fontsize=7,
                     alignment='left', labelspacing=.55)
    legend_ax.add_artist(score_legend)
    legend_ax.text(0, .735, 'Ceiling of the saved structure.\n'
                   'MC clipping uses a different\nrounded mass (see caption).',
                   transform=legend_ax.transAxes, va='top', fontsize=6.6,
                   linespacing=1.5, color=INK)
    legend_ax.legend(handles=[Line2D([], [], color='#444444', marker='o', ls='', ms=3.4,
                                     label='Mean OTC'),
                              Line2D([], [], color='#444444', marker='o', mfc='white',
                                     ls='', ms=3.4, label='Mean ceiling $C$')],
                     loc='upper left', bbox_to_anchor=(0, .565), frameon=False,
                     handlelength=1.1, title='Panel b', title_fontsize=7,
                     alignment='left', labelspacing=.55)
    legend_ax.text(0, .325,
                   'Panels d and e\n'
                   'Bars: mean of five runs.\nWhiskers: run minimum\nto maximum.\n'
                   'One scale per panel.\n\n'
                   'Reference cell: 12 A sites,\n24 B sites and 48 O sites.',
                   transform=legend_ax.transAxes, va='top', fontsize=6.6,
                   linespacing=1.5, color=INK)
    frame_panels(fig, [[[a] for a in panel_axes[:3]],
                       [[panel_axes[3]], [panel_axes[4]]]],
                 columns=[[panel_axes[0], panel_axes[3]], [panel_axes[1], panel_axes[4]]])

    save(fig, 'fig04_mechanism')
