import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIRS_AND_LABELS = {
    "Mistral English v1.1": {
        "path": "training/v1/training_logs/mistral_txt_sparql_en/",
        "color": "darkblue",
    },
    "Occiglot English v1.1": {
        "path": "training/v1/training_logs/occiglot_txt_sparql_en/",
        "color": "orange",
    },
    "Mistral German v1.1": {
        "path": "training/v1/training_logs/mistral_txt_sparql_de/",
        "color": "red",
    },
    "Occiglot German v1.1": {
        "path": "training/v1/training_logs/occiglot_txt_sparql_de/",
        "color": "deepskyblue",
    },
}
LOSS_TAG = "train/loss"
SMOOTHING_FACTOR_ALPHA = 0.6

OUTPUT_FILENAME = "train_loss_v1_1_with_legend.png"


def load_scalar_data(event_file_path, tag_name):
    """Loads scalar data for a given tag from a TensorBoard event file."""
    event_acc = EventAccumulator(event_file_path)
    event_acc.Reload()

    if tag_name not in event_acc.Tags()["scalars"]:
        print(
            f"Warning: Tag '{tag_name}' not found in {event_file_path}. Available tags: {event_acc.Tags()['scalars']}"
        )
        return [], []

    scalar_events = event_acc.Scalars(tag_name)
    steps = [event.step for event in scalar_events]
    values = [event.value for event in scalar_events]
    return steps, values


plt.figure(figsize=(12, 7))

for model_label, details in LOG_DIRS_AND_LABELS.items():
    log_dir = details["path"]
    event_file = None
    if os.path.isdir(log_dir):
        for f in os.listdir(log_dir):
            if "events.out.tfevents" in f:
                event_file = os.path.join(log_dir, f)
                break

    if not event_file:
        print(f"Warning: No event file found in {log_dir} for {model_label}. Skipping.")
        continue

    print(f"Loading data for {model_label} from {event_file} with tag '{LOSS_TAG}'...")
    steps, loss_values = load_scalar_data(event_file, LOSS_TAG)

    if not steps:
        print(f"No data loaded for {model_label}. Skipping.")
        continue

    if SMOOTHING_FACTOR_ALPHA is not None and SMOOTHING_FACTOR_ALPHA > 0:
        series = pd.Series(loss_values)
        smoothed_loss = (
            series.ewm(alpha=SMOOTHING_FACTOR_ALPHA, adjust=False).mean().tolist()
        )
    else:
        smoothed_loss = loss_values

    plt.plot(
        steps, smoothed_loss, label=model_label, color=details["color"], linewidth=1.5
    )

plt.xlabel("Training Steps", fontsize=12)
plt.ylabel(
    f"Training Loss (Smoothed, factor={SMOOTHING_FACTOR_ALPHA if SMOOTHING_FACTOR_ALPHA else 'None'})",
    fontsize=12,
)
plt.title("Training Loss Curves for v1.1 Models (Wikidata-only Subset)", fontsize=14)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend(fontsize=10)
plt.tight_layout()

try:
    plt.savefig(OUTPUT_FILENAME, dpi=300)
    print(f"Plot saved as {OUTPUT_FILENAME}")
except Exception as e:
    print(f"Error saving plot: {e}")
