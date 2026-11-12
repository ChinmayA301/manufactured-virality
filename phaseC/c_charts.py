"""Phase C figure: evasion cost vs the defender's false-flag rate, by defender design."""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, '..', 'results')
d = json.load(open(os.path.join(RES, 'c_github.json')))
FLOOR = min(d['settings']['COST'].values())
LAB = {'supervised': 'Supervised (timing + co-adoption)', 'monotone': 'Supervised, monotone',
       'anomaly_fpr1': 'Anomaly, 1% target', 'anomaly_fpr2': 'Anomaly, 2% target', 'anomaly_fpr5': 'Anomaly, 5% target',
       'anomaly_trimmed': 'Anomaly, trimmed reference'}
COL = {'supervised': '#4C72B0', 'monotone': '#8172B3', 'anomaly_fpr1': '#DD8452', 'anomaly_fpr2': '#E5A06E',
       'anomaly_fpr5': '#C44E52', 'anomaly_trimmed': '#55A868'}
fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
for ax, sc, title in ((axes[0], 'spam', 'Spam repo (campaign is the whole history)'), (axes[1], 'boost', 'Boosting a real repo')):
    for name in LAB:
        r = d['designs']['%s_%s' % (name, sc)]
        c = np.array(r['evasion_cost_per_star']) / FLOOR; x = 100 * r['untouched_target_flag_rate']
        ax.scatter([x] * len(c), c, color=COL[name], s=45, alpha=0.75, edgecolors='white', zorder=3)
        ax.plot([x - 0.25, x + 0.25], [np.median(c)] * 2, color=COL[name], lw=3, label='%s (naive dump caught %d%%)'
                % (LAB[name], round(100 * r['default_dump_detect'])))
    ax.axhline(1, color='#999', ls='--', lw=1)
    ax.set_yscale('log'); ax.set_xlabel('Untouched real repos flagged (%)'); ax.set_title(title, loc='left'); ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc='upper left')
axes[0].set_ylabel('Cheapest evasion found (cost per fake star, x cheapest account price)')
fig.suptitle('Phase C: what an optimising attacker pays to keep detection <= 10% (K=200 fake stars; 3 BO runs per design)', x=0.01, ha='left')
fig.text(0.01, -0.02, 'Simulation only. Relative account prices are assumptions (throwaway 1, farm 0.6, aged 8). Dots = BO runs, '
         're-evaluated on 120 fresh targets; bar = median. Dashed line = cheapest possible campaign.', fontsize=8, color='#666')
plt.tight_layout(); plt.savefig(os.path.join(RES, 'c_github.png'), dpi=130, bbox_inches='tight'); print('wrote c_github.png')
