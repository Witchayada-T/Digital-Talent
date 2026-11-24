"""Run the released CV evidence classifier on one requirement and CV excerpt."""

import argparse
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent / 'validation/demo/cv_evidence.joblib'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requirement', required=True, help='One skill or requirement from a job posting')
    parser.add_argument('--excerpt', required=True, help='A short CV excerpt to assess')
    args = parser.parse_args()

    requirement = args.requirement.strip()
    excerpt = args.excerpt.strip()
    if not requirement or not excerpt:
        parser.error('Enter both a job requirement and a CV excerpt.')

    pipeline = joblib.load(MODEL_PATH)
    example = pd.DataFrame({'requirement_text': [requirement], 'cv_excerpt': [excerpt]})
    print(pipeline.predict(example)[0])


if __name__ == '__main__':
    main()
