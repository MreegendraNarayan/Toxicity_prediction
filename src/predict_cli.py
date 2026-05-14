"""Command-line interface for true GNN toxicity predictions."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from toxicity.inference import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    MODEL_SPECS,
    TOX21_TASKS,
    InvalidSmilesError,
    ModelUnavailableError,
    ToxicityPredictor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def print_prediction(result):
    print(f"\nToxicity predictions for: {result['smiles']}")
    print(f"Model: {result['model']}")
    print("-" * 64)
    for task, probability in result["predictions"].items():
        bar_length = 20
        filled = int(bar_length * probability)
        bar = "#" * filled + "." * (bar_length - filled)
        print(f"  {task:<15} {probability:.4f} {bar}")
    print("-" * 64)
    print(f"  {'Average':<15} {result['average_toxicity']:.4f}")
    print(f"  {'Risk':<15} {result['risk_label']}")


def list_models(predictor: ToxicityPredictor) -> None:
    print("\nAvailable API models")
    print("-" * 64)
    for status in predictor.available_models():
        marker = "compatible" if status.compatible else "unavailable"
        size = f"{status.size_kb:.1f} KB" if status.size_kb is not None else "missing"
        print(f"  {status.name:<15} {marker:<12} {size:<12} {status.message}")
    print()


def run_batch(predictor: ToxicityPredictor, csv_path: str, output_path: str, model_name: str) -> None:
    input_path = Path(csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(input_path)
    if "smiles" not in df.columns:
        raise ValueError("CSV must have a 'smiles' column")

    output_rows = []
    errors = []
    for idx, row in df.iterrows():
        try:
            result = predictor.predict_one(str(row["smiles"]), model_name)
            output_rows.append(
                {
                    **row.to_dict(),
                    **result["predictions"],
                    "average_toxicity": result["average_toxicity"],
                    "risk_label": result["risk_label"],
                }
            )
        except InvalidSmilesError as exc:
            errors.append({"row": idx, "smiles": row["smiles"], "error": str(exc)})

    if not output_rows:
        raise RuntimeError("No valid SMILES strings found in CSV")

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_path, index=False)
    print(f"\nBatch prediction complete: {len(output_rows)} rows written to {output_path}")
    if errors:
        print(f"Skipped {len(errors)} invalid rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Toxicity Prediction CLI")
    parser.add_argument("--smiles", help="SMILES string for single prediction")
    parser.add_argument("--csv", help="CSV file for batch prediction; must contain a smiles column")
    parser.add_argument("--output", default="toxicity_predictions.csv", help="Output CSV path")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--checkpoint-dir", default="models/checkpoints")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    predictor = ToxicityPredictor(checkpoint_dir=args.checkpoint_dir)

    if args.list_models:
        list_models(predictor)
        return

    if not args.smiles and not args.csv:
        parser.print_help()
        sys.exit(1)

    try:
        if args.smiles:
            print_prediction(predictor.predict_one(args.smiles, args.model))

        if args.csv:
            run_batch(predictor, args.csv, args.output, args.model)
    except (InvalidSmilesError, ModelUnavailableError, FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
