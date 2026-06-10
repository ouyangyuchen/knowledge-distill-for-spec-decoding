# %% [markdown]
# # Qwen3 Runtime Sweep Heatmap
#
# Visualize a gamma x temperature heatmap from the runtime sweep CSV.
# The plot is fixed to `max_new_tokens == 256`.

# %%
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CSV_PATH = Path(
    "/scratch/cs552-results/"
    "qwen3_runtime_sweep_qwen3_8btarget_0p6b_tgen_jsd_"
    "ultrachat_50k_target_gen_seed42.csv"
)
MAX_NEW_TOKENS = 256
SAVE_PLOT_PATH = Path("qwen3_runtime_heatmap.png")

# Change this to "acceptance_rate", "avg_accepted_tokens", "tokens_per_second",
# or another numeric column from the CSV.
METRIC = "speedup"


# %%
def load_metric_grid(
    csv_path: Path,
    *,
    metric: str,
    max_new_tokens: int,
) -> tuple[list[int], list[float], np.ndarray]:
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") != "ok":
                continue
            if int(row["max_new_tokens"]) != max_new_tokens:
                continue
            rows.append(row)

    if not rows:
        raise ValueError(f"No successful rows found with max_new_tokens={max_new_tokens}")

    missing = [name for name in ("gamma", "runtime_temperature", metric) if name not in rows[0]]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    gammas = sorted({int(row["gamma"]) for row in rows})
    temperatures = sorted({float(row["runtime_temperature"]) for row in rows})

    gamma_idx = {value: idx for idx, value in enumerate(gammas)}
    temp_idx = {value: idx for idx, value in enumerate(temperatures)}
    grid = np.full((len(gammas), len(temperatures)), np.nan, dtype=float)

    for row in rows:
        gamma = int(row["gamma"])
        temperature = float(row["runtime_temperature"])
        grid[gamma_idx[gamma], temp_idx[temperature]] = float(row[metric])

    return gammas, temperatures, grid


gammas, temperatures, values = load_metric_grid(
    CSV_PATH,
    metric=METRIC,
    max_new_tokens=MAX_NEW_TOKENS,
)

print(f"Loaded {CSV_PATH}")
print(f"Metric: {METRIC}")
print(f"max_new_tokens: {MAX_NEW_TOKENS}")
print(f"gamma values: {gammas}")
print(f"temperature values: {temperatures}")


# %%
fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)

masked_values = np.ma.masked_invalid(values)
image = ax.imshow(masked_values, cmap="viridis", aspect="auto")

ax_label_size = 18

ax.set_title(f"{METRIC.replace('_', ' ').title()} Heatmap (max_new_tokens={MAX_NEW_TOKENS})")
ax.set_xlabel("Runtime temperature", fontsize=ax_label_size)
ax.set_ylabel("Gamma", fontsize=ax_label_size)
ax.set_xticks(np.arange(len(temperatures)), labels=[f"{temp:g}" for temp in temperatures])
ax.set_yticks(np.arange(len(gammas)), labels=[str(gamma) for gamma in gammas])

cbar = fig.colorbar(image, ax=ax)
cbar.set_label(METRIC.replace("_", " ").title(), fontsize=ax_label_size)

for row_idx, gamma in enumerate(gammas):
    for col_idx, temperature in enumerate(temperatures):
        value = values[row_idx, col_idx]
        if np.isnan(value):
            label = "NA"
        elif abs(value) >= 100:
            label = f"{value:.0f}"
        elif abs(value) >= 10:
            label = f"{value:.1f}"
        else:
            label = f"{value:.2f}"

        ax.text(
            col_idx,
            row_idx,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=16,
        )

plt.savefig(SAVE_PLOT_PATH, dpi=300)
print(f"Saved heatmap to {SAVE_PLOT_PATH}")


# %%
best_idx = np.unravel_index(np.nanargmax(values), values.shape)
best_gamma = gammas[best_idx[0]]
best_temperature = temperatures[best_idx[1]]
best_value = values[best_idx]

print(
    f"Best {METRIC} at max_new_tokens={MAX_NEW_TOKENS}: "
    f"gamma={best_gamma}, temperature={best_temperature:g}, {METRIC}={best_value:.4g}"
)

