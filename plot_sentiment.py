#!/usr/bin/env python3
"""
Plot sentiment vs market results.
Generates a multi-panel figure saved as sentiment_plots.png
"""

import csv
import re
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

BASE_DIR = "/Users/prerak/Desktop/CRYPTO"
RESULTS_CSV = f"{BASE_DIR}/sentiment_results.csv"
COMPARE_CSV = f"{BASE_DIR}/sentiment_vs_market.csv"
OUTPUT_PNG = f"{BASE_DIR}/sentiment_plots.png"

# ── Load sentiment results ──
dates, sentiments, sp500_pcts, directions = [], [], [], []

with open(RESULTS_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        d = row.get("market_date") or ""
        if not d:
            continue
        dt = datetime.datetime.strptime(d, "%Y-%m-%d")
        pct = row.get("sp500_change_pct", "")
        if not pct:
            continue
        dates.append(dt)
        sentiments.append(row["sentiment"])
        sp500_pcts.append(float(pct))
        directions.append(row["market_direction"])

# ── Load accuracy summary from compare CSV ──
tickers, accuracies = [], []
pos_accs, neg_accs, neu_accs = [], [], []
in_summary = False

with open(COMPARE_CSV) as f:
    reader = csv.reader(f)
    for row in reader:
        if row and row[0] == "Ticker":
            in_summary = True
            continue
        if in_summary and len(row) >= 7 and row[0] != "=== ACCURACY SUMMARY ===":
            tickers.append(row[0])
            accuracies.append(float(row[3]))
            pos_accs.append(float(row[4]))
            neg_accs.append(float(row[5]))
            neu_accs.append(float(row[6]))

# ── Color setup ──
COLORS = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
plt.style.use("seaborn-v0_8-darkgrid")
fig = plt.figure(figsize=(20, 24))
fig.suptitle("WSJ Front Page Sentiment vs Stock Market Performance",
             fontsize=22, fontweight="bold", y=0.98)

# ═══════════════════════════════════════════════
# PLOT 1: S&P 500 daily returns colored by sentiment
# ═══════════════════════════════════════════════
ax1 = fig.add_subplot(4, 2, 1)
for dt, pct, sent in zip(dates, sp500_pcts, sentiments):
    ax1.bar(dt, pct, color=COLORS.get(sent, "gray"), width=1.5, alpha=0.7, edgecolor="none")

ax1.axhline(0, color="black", linewidth=0.5)
ax1.set_title("S&P 500 Daily Returns Colored by Sentiment", fontsize=13, fontweight="bold")
ax1.set_ylabel("Daily Return (%)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLORS[s], label=s.capitalize()) for s in ["positive", "negative", "neutral"]]
ax1.legend(handles=legend_elements, loc="upper left", fontsize=9)

# ═══════════════════════════════════════════════
# PLOT 2: Accuracy by ticker (horizontal bar)
# ═══════════════════════════════════════════════
ax2 = fig.add_subplot(4, 2, 2)
sorted_idx = np.argsort(accuracies)
y_pos = np.arange(len(tickers))
bars = ax2.barh(y_pos, [accuracies[i] for i in sorted_idx], color="#3498db", alpha=0.8, edgecolor="white")
ax2.set_yticks(y_pos)
ax2.set_yticklabels([tickers[i] for i in sorted_idx], fontsize=10)
ax2.set_xlabel("Accuracy (%)")
ax2.set_title("Sentiment Prediction Accuracy by Ticker", fontsize=13, fontweight="bold")
ax2.axvline(50, color="red", linestyle="--", linewidth=1, label="Coin flip (50%)")
ax2.legend(fontsize=9)
for bar, idx in zip(bars, sorted_idx):
    ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
             f"{accuracies[idx]:.1f}%", va="center", fontsize=9, fontweight="bold")

# ═══════════════════════════════════════════════
# PLOT 3: Sentiment distribution pie chart
# ═══════════════════════════════════════════════
ax3 = fig.add_subplot(4, 2, 3)
sent_counts = {s: sentiments.count(s) for s in ["positive", "negative", "neutral"]}
labels = [f"{k.capitalize()}\n({v})" for k, v in sent_counts.items()]
colors = [COLORS[k] for k in sent_counts]
wedges, texts, autotexts = ax3.pie(sent_counts.values(), labels=labels, colors=colors,
                                    autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
for at in autotexts:
    at.set_fontweight("bold")
ax3.set_title("Sentiment Distribution", fontsize=13, fontweight="bold")

# ═══════════════════════════════════════════════
# PLOT 4: Market direction when sentiment is positive vs negative
# ═══════════════════════════════════════════════
ax4 = fig.add_subplot(4, 2, 4)

# When sentiment = positive, how often was market up/down?
pos_up = sum(1 for s, d in zip(sentiments, directions) if s == "positive" and d == "up")
pos_down = sum(1 for s, d in zip(sentiments, directions) if s == "positive" and d == "down")
pos_flat = sum(1 for s, d in zip(sentiments, directions) if s == "positive" and d == "flat")

neg_up = sum(1 for s, d in zip(sentiments, directions) if s == "negative" and d == "up")
neg_down = sum(1 for s, d in zip(sentiments, directions) if s == "negative" and d == "down")
neg_flat = sum(1 for s, d in zip(sentiments, directions) if s == "negative" and d == "flat")

x = np.arange(3)
width = 0.35
bars1 = ax4.bar(x - width / 2, [pos_up, pos_down, pos_flat], width,
                label="Positive Sentiment", color="#2ecc71", alpha=0.8)
bars2 = ax4.bar(x + width / 2, [neg_up, neg_down, neg_flat], width,
                label="Negative Sentiment", color="#e74c3c", alpha=0.8)
ax4.set_xticks(x)
ax4.set_xticklabels(["Market Up", "Market Down", "Market Flat"], fontsize=11)
ax4.set_ylabel("Number of Days")
ax4.set_title("Actual Market Direction by Sentiment Call", fontsize=13, fontweight="bold")
ax4.legend(fontsize=10)
ax4.bar_label(bars1, fontsize=9, fontweight="bold")
ax4.bar_label(bars2, fontsize=9, fontweight="bold")

# ═══════════════════════════════════════════════
# PLOT 5: Accuracy breakdown (pos/neg/neutral) per ticker - grouped bar
# ═══════════════════════════════════════════════
ax5 = fig.add_subplot(4, 2, (5, 6))
x = np.arange(len(tickers))
width = 0.22
bars_p = ax5.bar(x - width, pos_accs, width, label="When Positive", color="#2ecc71", alpha=0.8)
bars_n = ax5.bar(x, neg_accs, width, label="When Negative", color="#e74c3c", alpha=0.8)
bars_u = ax5.bar(x + width, neu_accs, width, label="When Neutral", color="#95a5a6", alpha=0.8)
ax5.set_xticks(x)
ax5.set_xticklabels(tickers, rotation=35, ha="right", fontsize=9)
ax5.set_ylabel("Accuracy (%)")
ax5.set_title("Prediction Accuracy by Sentiment Type & Ticker", fontsize=13, fontweight="bold")
ax5.legend(fontsize=10)
ax5.axhline(50, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

# ═══════════════════════════════════════════════
# PLOT 6: Rolling 7-day accuracy (S&P 500)
# ═══════════════════════════════════════════════
ax6 = fig.add_subplot(4, 2, 7)
# Calculate match array for S&P 500
matches = []
for s, d in zip(sentiments, directions):
    if s == "positive" and d == "up":
        matches.append(1)
    elif s == "negative" and d == "down":
        matches.append(1)
    elif s == "neutral" and d == "flat":
        matches.append(1)
    else:
        matches.append(0)

window = 14
if len(matches) >= window:
    rolling_acc = [np.mean(matches[max(0, i - window + 1):i + 1]) * 100 for i in range(len(matches))]
    ax6.plot(dates, rolling_acc, color="#3498db", linewidth=1.5, alpha=0.8)
    ax6.fill_between(dates, rolling_acc, alpha=0.15, color="#3498db")
    ax6.axhline(50, color="red", linestyle="--", linewidth=1, alpha=0.5, label="50% (coin flip)")
    ax6.axhline(np.mean(matches) * 100, color="orange", linestyle="--", linewidth=1, alpha=0.7,
                label=f"Overall avg: {np.mean(matches)*100:.1f}%")

ax6.set_title(f"Rolling {window}-Day Accuracy (S&P 500)", fontsize=13, fontweight="bold")
ax6.set_ylabel("Accuracy (%)")
ax6.set_ylim(0, 100)
ax6.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax6.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax6.legend(fontsize=9)

# ═══════════════════════════════════════════════
# PLOT 7: Average return by sentiment
# ═══════════════════════════════════════════════
ax7 = fig.add_subplot(4, 2, 8)
avg_returns = {}
for sent_type in ["positive", "negative", "neutral"]:
    returns = [p for s, p in zip(sentiments, sp500_pcts) if s == sent_type]
    avg_returns[sent_type] = np.mean(returns) if returns else 0

bars = ax7.bar(
    [s.capitalize() for s in avg_returns],
    list(avg_returns.values()),
    color=[COLORS[s] for s in avg_returns],
    alpha=0.8,
    edgecolor="white",
    width=0.5,
)
ax7.axhline(0, color="black", linewidth=0.5)
ax7.set_ylabel("Average S&P 500 Return (%)")
ax7.set_title("Avg S&P 500 Return by Sentiment", fontsize=13, fontweight="bold")
for bar, val in zip(bars, avg_returns.values()):
    ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:+.3f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved plots to {OUTPUT_PNG}")
plt.close()
