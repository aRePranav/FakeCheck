"""
main.py
-------
FakeCheck — Full Pipeline Runner
----------------------------------
Author  : R Pranav
Project : Spam News Detection | Intrainz AI Internship

Runs the complete pipeline:
  1. Load + clean WELFake dataset
  2. TF-IDF feature extraction
  3. Train + evaluate 4 ML models
  4. Generate all visualizations
  5. Save best model for inference
  6. Run live prediction demo

Usage:
  python main.py
  python main.py --data data/WELFake_Dataset.csv
  python main.py --predict "Your news headline here"
"""

import os
import sys
import argparse
import joblib

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess    import load_and_prepare
from vectorize     import build_tfidf
from train_evaluate import (
    train_and_evaluate_all,
    print_summary_table,
    plot_confusion_matrices,
    plot_model_comparison,
)
from fakecheck     import FakeCheck
from visualize     import generate_all_plots


DEMO_HEADLINES = [
    "NASA confirms water ice discovered on the Moon surface near polar craters",
    "SHOCKING: Government secretly poisoning water supply to control population!",
    "Federal Reserve raises interest rates by 25 basis points citing inflation data",
    "You WILL NOT believe what this celebrity did! Doctors are completely SPEECHLESS!",
    "Scientists publish new study linking sleep deprivation to memory loss in adults",
    "BREAKING: Deep state plans revealed — they have been lying to us all along!!!",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="FakeCheck — Real-Time Fake News Detection System"
    )
    parser.add_argument(
        "--data",
        type    = str,
        default = "data/WELFake_Dataset.csv",
        help    = "Path to WELFake_Dataset.csv (default: data/WELFake_Dataset.csv)",
    )
    parser.add_argument(
        "--predict",
        type    = str,
        default = None,
        help    = "Skip training and predict a single article text directly",
    )
    parser.add_argument(
        "--model_dir",
        type    = str,
        default = "models",
        help    = "Directory to save/load trained models (default: models/)",
    )
    parser.add_argument(
        "--output_dir",
        type    = str,
        default = "outputs",
        help    = "Directory to save plots (default: outputs/)",
    )
    return parser.parse_args()


def run_training_pipeline(args):
    """Full training pipeline from raw data to saved models."""

    print("=" * 65)
    print("  FakeCheck — Spam News Detection System")
    print("  Author: R Pranav | Intrainz AI Internship")
    print("=" * 65)

    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_prepare(args.data)

    # Step 2: TF-IDF vectorization
    vectorizer_path = os.path.join(args.model_dir, "tfidf_vectorizer.pkl")
    X_tr_tfidf, X_te_tfidf, vectorizer = build_tfidf(
        X_train, X_test,
        save_path = vectorizer_path,
    )

    # Step 3: Train and evaluate all models
    results, best_name = train_and_evaluate_all(
        X_tr_tfidf, X_te_tfidf,
        y_train, y_test,
        output_dir = args.model_dir,
    )

    # Step 4: Print summary table
    print_summary_table(results)

    # Step 5: Copy best model to a standard filename for FakeCheck class
    best_safe   = best_name.lower().replace(" ", "_")
    best_src    = os.path.join(args.model_dir, f"{best_safe}.pkl")
    best_dst    = os.path.join(args.model_dir, "passive_aggressive.pkl")
    if not os.path.exists(best_dst):
        import shutil
        shutil.copy(best_src, best_dst)
    print(f"\n  Production model: {best_name}")
    print(f"  Saved as        : {best_dst}")

    # Step 6: Visualizations
    best_model = results[best_name]["model"]
    generate_all_plots(
        results, vectorizer, best_model,
        y_train, y_test,
        output_dir = args.output_dir,
    )

    return results, best_name


def run_demo(model_dir: str = "models"):
    """Run live prediction demo on sample headlines."""
    checker = FakeCheck(
        model_path      = os.path.join(model_dir, "passive_aggressive.pkl"),
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl"),
    )

    print("\n" + "=" * 65)
    print("  FAKECHECK LIVE DEMO")
    print("=" * 65)

    for headline in DEMO_HEADLINES:
        checker.predict_verbose(headline)


def main():
    args = parse_args()

    # If --predict flag is used, skip training and predict directly
    if args.predict:
        model_path = os.path.join(args.model_dir, "passive_aggressive.pkl")
        vec_path   = os.path.join(args.model_dir, "tfidf_vectorizer.pkl")

        if not os.path.exists(model_path) or not os.path.exists(vec_path):
            print("No trained model found. Running full training pipeline first...\n")
            run_training_pipeline(args)

        checker = FakeCheck(model_path=model_path, vectorizer_path=vec_path)
        checker.predict_verbose(args.predict)
        return

    # Full pipeline
    run_training_pipeline(args)
    run_demo(model_dir=args.model_dir)

    print("\n" + "=" * 65)
    print("  Pipeline complete.")
    print(f"  Models saved in : {os.path.abspath(args.model_dir)}/")
    print(f"  Plots saved in  : {os.path.abspath(args.output_dir)}/")
    print("\n  To predict any article:")
    print('  python main.py --predict "Your news headline here"')
    print("=" * 65)


if __name__ == "__main__":
    main()
