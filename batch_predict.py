"""
batch_predict.py
----------------
Classify an entire CSV file of news articles using FakeCheck.
Results saved to outputs/predictions.csv with label + confidence columns.

Usage:
  python batch_predict.py --input data/articles.csv --text_col text
  python batch_predict.py --input data/articles.csv --text_col content --output outputs/results.csv

Author: R Pranav | Intrainz AI Internship
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from fakecheck import FakeCheck


def parse_args():
    parser = argparse.ArgumentParser(description="FakeCheck Batch Predictor")
    parser.add_argument("--input",     type=str, required=True,  help="Path to input CSV file")
    parser.add_argument("--text_col",  type=str, default="text", help="Column name containing news text")
    parser.add_argument("--output",    type=str, default="outputs/predictions.csv", help="Path to save results CSV")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory with trained models")
    return parser.parse_args()


def batch_predict(input_csv, text_col, output_csv, model_dir="models"):
    """
    Load a CSV, classify every row, and save results.

    Adds columns:
        Prediction   : FAKE or REAL
        Confidence_% : calibrated confidence score
    """
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)

    print(f"Loading input file: {input_csv}")
    df = pd.read_csv(input_csv)

    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found in CSV.\n"
            f"Available columns: {list(df.columns)}"
        )

    print(f"Total articles to classify: {len(df)}")

    checker = FakeCheck(
        model_path      = os.path.join(model_dir, "passive_aggressive.pkl"),
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl"),
    )

    predictions  = []
    confidences  = []

    for i, text in enumerate(df[text_col].fillna("").astype(str)):
        result = checker.predict(text)
        predictions.append(result["label"])
        confidences.append(result["confidence_pct"])

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(df)} articles...")

    df["Prediction"]    = predictions
    df["Confidence_%"]  = confidences

    df.to_csv(output_csv, index=False)

    # Summary
    print(f"\nResults saved to: {output_csv}")
    print("\nPrediction Summary:")
    print(df["Prediction"].value_counts().to_string())
    print(f"\nMean Confidence: {df['Confidence_%'].mean():.2f}%")


if __name__ == "__main__":
    args = parse_args()
    batch_predict(
        input_csv  = args.input,
        text_col   = args.text_col,
        output_csv = args.output,
        model_dir  = args.model_dir,
    )
