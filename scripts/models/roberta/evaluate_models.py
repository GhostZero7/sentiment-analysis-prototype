from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_models import evaluate_models


def main() -> int:
    results = evaluate_models(
        PROJECT_ROOT / "data" / "processed" / "labeled" / "testing" / "final_label_roberta_test.csv",
        label_column="roberta_label",
        models_dir=PROJECT_ROOT / "data" / "models" / "roberta",
        results_dir=PROJECT_ROOT / "data" / "results" / "roberta",
    )
    print(f"Evaluation summary saved to {results['summary_path']}")
    print(f"Classification reports saved to {results['reports_path']}")
    print(f"Confusion matrices saved in {results['confusion_matrix_dir']}")
    for row in results["summary"]:
        print(
            f"{row['model']}: accuracy={row['accuracy']:.4f}, "
            f"f1_weighted={row['f1_weighted']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
