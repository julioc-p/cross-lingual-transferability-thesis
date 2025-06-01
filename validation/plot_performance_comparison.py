import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os


def parse_summary_file(file_path):
    """
    Parses a summary.txt file to extract Macro Precision, Recall, F1-score,
    and Executable Query Percentage.
    """
    metrics = {
        "precision": None,
        "recall": None,
        "f1": None,
        "executable_percentage": None,
    }
    try:
        if not os.path.exists(file_path):
            print(f"Warning: File not found during parsing - {file_path}")
            return None

        with open(file_path, "r") as f:
            content = f.read()

        precision_match = re.search(r"Macro Precision:\s*([\d.]+)", content)
        recall_match = re.search(r"Macro Recall:\s*([\d.]+)", content)
        f1_match = re.search(r"Macro F1-Score:\s*([\d.]+)", content)
        executable_match = re.search(
            r"Generated Query Non-Empty & Executable:\s*\d+\s*\(\s*([\d.]+)\s*%\)",
            content,
        )

        if precision_match:
            metrics["precision"] = float(precision_match.group(1))
        if recall_match:
            metrics["recall"] = float(recall_match.group(1))
        if f1_match:
            metrics["f1"] = float(f1_match.group(1))
        if executable_match:
            metrics["executable_percentage"] = float(executable_match.group(1))

        if all(v is None for v in metrics.values()):
            return None
        if any(
            v is None
            for v in [
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["executable_percentage"],
            ]
        ):
            print(
                f"Warning: Some macro metrics or executable percentage might be missing in {file_path}. Found: {metrics}"
            )
        return metrics
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None


def main():
    data = []
    patterns = [
        {
            "regex": r"^(QALD10)/results/(v[12])/(baseline|finetuned)/(mistral|occiglot)/(en|de)/summary\.txt$",
            "groups": {
                "dataset_raw": 1,
                "version": 2,
                "training": 3,
                "model": 4,
                "lang": 5,
            },
            "dataset_name_override": None,
            "training_status_override": None,
        },
        {
            "regex": r"^(test_set)/(v1)/(mistral|occiglot)/(en|de)/(baseline|finetuned)/summary\.txt$",
            "groups": {
                "dataset_raw": 1,
                "version": 2,
                "model": 3,
                "lang": 4,
                "training": 5,
            },
            "dataset_name_override": "Test Set",
            "training_status_override": None,
        },
        {
            "regex": r"^(test_set)/(v2)/results/(mistral|occiglot)/(en|de)/(baseline|finetuned)/summary\.txt$",
            "groups": {
                "dataset_raw": 1,
                "version": 2,
                "model": 3,
                "lang": 4,
                "training": 5,
            },
            "dataset_name_override": "Test Set",
            "training_status_override": None,
        },
    ]

    found_files_count = 0
    parsed_files_count = 0
    unrecognized_paths = []

    print("--- Starting File Discovery (from current directory) ---")
    for root, _, filenames in os.walk("."):
        for filename in filenames:
            if filename == "summary.txt":
                found_files_count += 1
                file_path = os.path.join(root, filename)
                relative_path = os.path.normpath(file_path)

                matched_pattern = False
                for p_info in patterns:
                    match = re.match(p_info["regex"], relative_path, re.IGNORECASE)
                    if match:
                        matched_pattern = True
                        parsed_info = {}

                        raw_dataset_name = match.group(p_info["groups"]["dataset_raw"])
                        parsed_info["dataset_name"] = p_info[
                            "dataset_name_override"
                        ] or (
                            "Test Set"
                            if raw_dataset_name.lower() == "test_set"
                            else raw_dataset_name.capitalize()
                        )

                        parsed_info["dataset_version"] = match.group(
                            p_info["groups"]["version"]
                        )
                        parsed_info["model_type"] = match.group(
                            p_info["groups"]["model"]
                        ).capitalize()
                        parsed_info["language"] = match.group(
                            p_info["groups"]["lang"]
                        ).lower()

                        if p_info["training_status_override"]:
                            parsed_info["training_status"] = p_info[
                                "training_status_override"
                            ]
                        elif "training" in p_info["groups"]:
                            parsed_info["training_status"] = match.group(
                                p_info["groups"]["training"]
                            ).capitalize()
                        else:
                            print(
                                f"Warning: Training status could not be determined for {relative_path} with pattern {p_info['regex']}. Skipping."
                            )
                            continue

                        metrics = parse_summary_file(file_path)
                        if (
                            metrics
                            and metrics.get("f1") is not None
                            and metrics.get("precision") is not None
                            and metrics.get("recall") is not None
                            and metrics.get("executable_percentage") is not None
                        ):
                            parsed_files_count += 1
                            data.append(
                                {
                                    "Dataset": f"{parsed_info['dataset_name']} {parsed_info['dataset_version']}",
                                    "Model": parsed_info["model_type"],
                                    "Training": parsed_info["training_status"],
                                    "Language": parsed_info["language"].upper(),
                                    "Precision": metrics["precision"],
                                    "Recall": metrics["recall"],
                                    "F1-Score": metrics["f1"],
                                    "Executable Queries (%)": metrics[
                                        "executable_percentage"
                                    ],
                                    "Category": f"{parsed_info['dataset_name']} {parsed_info['dataset_version']} - {parsed_info['language'].upper()} - {parsed_info['training_status']}",
                                    "_source_path": relative_path,
                                }
                            )
                        elif metrics:
                            print(
                                f"Note: Metrics parsed for {file_path} but F1, Precision, Recall, or Executable Queries (%) might be missing/invalid. Metrics: {metrics}. Skipping."
                            )
                        else:
                            pass
                        break

                if not matched_pattern and os.path.exists(file_path):
                    unrecognized_paths.append(relative_path)

    print(f"\n--- File Discovery Summary ---")
    print(f"Total 'summary.txt' files found: {found_files_count}")
    print(
        f"Files with valid metrics successfully parsed for plotting: {parsed_files_count}"
    )
    if unrecognized_paths:
        print(
            f"Files named 'summary.txt' found but their path structure was not recognized ({len(unrecognized_paths)}):"
        )
        for path_str in unrecognized_paths:
            print(f"  - {path_str}")
    print("----------------------------\n")

    if not data:
        print(
            "No data to plot. Please check file paths, content, and recognition patterns."
        )
        return

    df = pd.DataFrame(data)

    if df.empty:
        print("DataFrame is empty after processing. No valid data found for plotting.")
        return

    df = df.sort_values(by=["Dataset", "Language", "Training", "Model"])


    df_qald = df[df["Dataset"].str.startswith("Qald10")].copy()
    df_test_v1 = df[df["Dataset"] == "Test Set v1"].copy()
    df_test_v2 = df[df["Dataset"] == "Test Set v2"].copy()

    if not df_qald.empty:
        df_qald.loc[:, "SubCategory"] = (
            df_qald["Language"]
            + " - "
            + df_qald["Training"]
            + " ("
            + df_qald["Dataset"].apply(lambda x: x.split()[-1])
            + ")"
        )
    if not df_test_v1.empty:
        df_test_v1.loc[:, "SubCategory"] = (
            df_test_v1["Language"] + " - " + df_test_v1["Training"]
        )
    if not df_test_v2.empty:
        df_test_v2.loc[:, "SubCategory"] = (
            df_test_v2["Language"] + " - " + df_test_v2["Training"]
        )

    metrics_to_plot = ["F1-Score", "Precision", "Recall", "Executable Queries (%)"]

    for metric in metrics_to_plot:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

        ax_qald = fig.add_subplot(gs[0, :])
        ax_test_v1 = fig.add_subplot(gs[1, 0])
        ax_test_v2 = fig.add_subplot(gs[1, 1])

        subplot_config = [
            (df_qald, ax_qald, "QALD10 Results"),
            (df_test_v1, ax_test_v1, "Test Set v1 Results"),
            (df_test_v2, ax_test_v2, "Test Set v2 Results"),
        ]

        figure_handles, figure_labels = [], []

        for sub_df, ax, title_prefix in subplot_config:
            if sub_df.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for\n{title_prefix}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=12,
                    transform=ax.transAxes,
                )
                ax.set_title(f"{title_prefix}\n({metric})", fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            sns.barplot(
                x="SubCategory",
                y=metric,
                hue="Model",
                data=sub_df,
                ax=ax,
                palette="viridis",
                errorbar=None,
            )
            ax.set_title(f"{title_prefix}\n({metric})", fontsize=14)
            ax.set_xlabel("Lang - Training", fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.tick_params(axis="x", rotation=45, labelsize=9)
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")
            ax.tick_params(axis="y", labelsize=9)

            for p_patch in ax.patches:
                ax.annotate(
                    f"{p_patch.get_height():.3f}",
                    (p_patch.get_x() + p_patch.get_width() / 2.0, p_patch.get_height()),
                    ha="center",
                    va="center",
                    xytext=(0, 5),
                    textcoords="offset points",
                    fontsize=7,
                )

            if not figure_handles:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    figure_handles.extend(handles)
                    figure_labels.extend(labels)

            if ax.get_legend() is not None:
                ax.get_legend().remove()

        if figure_handles:
            unique_legend_items = dict(zip(figure_labels, figure_handles))
            fig.legend(
                unique_legend_items.values(),
                unique_legend_items.keys(),
                loc="upper right",
                bbox_to_anchor=(0.99, 0.98),
                title="Model",
                fontsize=10,
            )

        fig.suptitle(f"{metric} Comparison: Mistral vs. Occiglot", fontsize=18, y=0.995)
        plt.tight_layout(
            rect=[0, 0.03, 0.95, 0.95]
        )

        output_filename_metric_part = metric.lower().replace("-", "_").replace(" ", "_")
        output_filename_metric_part = re.sub(r"[^\w_]", "", output_filename_metric_part)
        output_filename = f"{output_filename_metric_part}_comparison_subplots.png"
        try:
            plt.savefig(output_filename, dpi=300)
            print(f"Saved plot: {output_filename}")
        except Exception as e:
            print(f"Error saving plot {output_filename}: {e}")
        plt.show()


    print("\n--- Summary of Data Used for Plotting ---")
    print(df)


if __name__ == "__main__":
    main()
