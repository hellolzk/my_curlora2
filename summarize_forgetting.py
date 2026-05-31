import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize continual-learning forgetting metrics.")
    parser.add_argument("--metrics_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--metric_priority",
        nargs="+",
        default=["label_accuracy", "rougeL", "exact_match"],
        help="Metric names in priority order for each task.",
    )
    return parser.parse_args()


def choose_metric(row, metric_priority):
    for metric in metric_priority:
        if metric in row:
            return metric, float(row[metric])
    raise KeyError(f"No usable metric found in row: {row}")


def main():
    args = parse_args()
    metrics_path = Path(args.metrics_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    by_task = defaultdict(list)
    for row in rows:
        metric_name, score = choose_metric(row, args.metric_priority)
        by_task[row["task"]].append(
            {
                "stage": row["stage"],
                "task": row["task"],
                "metric": metric_name,
                "score": score,
            }
        )

    summary = []
    for task, task_rows in by_task.items():
        best_score = max(item["score"] for item in task_rows)
        final_score = task_rows[-1]["score"]
        first_score = task_rows[0]["score"]
        summary.append(
            {
                "task": task,
                "metric": task_rows[-1]["metric"],
                "first_stage": task_rows[0]["stage"],
                "final_stage": task_rows[-1]["stage"],
                "first_score": first_score,
                "best_score": best_score,
                "final_score": final_score,
                "forgetting_best_minus_final": best_score - final_score,
                "drop_first_minus_final": first_score - final_score,
            }
        )

    with open(output_dir / "forgetting_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(output_dir / "forgetting_summary.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "task",
            "metric",
            "first_stage",
            "final_stage",
            "first_score",
            "best_score",
            "final_score",
            "forgetting_best_minus_final",
            "drop_first_minus_final",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved forgetting summary to {output_dir}")


if __name__ == "__main__":
    main()
