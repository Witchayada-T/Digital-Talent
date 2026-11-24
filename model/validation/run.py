"""Generate or read requirement-level labels, tune on validation, and evaluate on test."""
import argparse
import json
from itertools import product
from pathlib import Path
import random
import re

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

LABELS = ['evidence found', 'no evidence found', 'unclear']
SPLITS = ['train', 'validation', 'test']
SKILLS = {
    'SQL': ['queried relational databases', 'joined database tables'],
    'data analysis': ['examined operational datasets', 'interpreted performance measures'],
    'data visualization': ['designed dashboards', 'built graphical reports'],
    'financial analysis': ['reviewed budget variances', 'compared revenue forecasts'],
    'clinical reporting': ['summarized patient metrics', 'prepared hospital indicators'],
    'quality analysis': ['investigated defects', 'measured production variation'],
}
INDUSTRY_SKILL = {
    'Finance and insurance': 'financial analysis',
    'Health care and social assistance': 'clinical reporting',
    'Manufacturing': 'quality analysis',
}
# Each split uses different sentence frames. These are generated examples, with
# labels supplied by the meaning of the templates.
FRAMES = {
    'train': [
        ['I used {skill} on a project.', 'I independently {action}.'],
        ['I have no experience with {skill}.', 'My work was limited to scheduling meetings.'],
        ['I am interested in learning {skill}.', 'My team did {skill}, but my role was administrative.'],
    ],
    'validation': [
        ['My responsibilities included {skill}.', 'In my last role I {action}.'],
        ['I did not use {skill} in my previous role.', 'I maintained mailing lists and booked rooms.'],
        ['I shadowed a colleague who handled {skill}.', 'I plan to take a course in {skill}.'],
    ],
    'test': [
        ['I delivered a project that required {skill}.', 'I {action} for the department.'],
        ['I have never performed {skill}.', 'I organized travel and answered calls.'],
        ['My training introduced {skill} without a practical assignment.', 'I supported a team that used {skill}.'],
    ],
}


def make_demo():
    rng = random.Random(42)
    rows = []
    shared = ['SQL', 'data analysis', 'data visualization']
    for split, pairs in zip(SPLITS, [90, 30, 30]):
        for industry, index in product(INDUSTRY_SKILL, range(pairs // 3)):
            pair_id = f'{split}-{industry}-{index:02}'
            skills = [*shared, INDUSTRY_SKILL[industry]]
            labels = [*LABELS, LABELS[index % 3]]
            rng.shuffle(labels)
            for skill, label in zip(skills, labels):
                frame = rng.choice(FRAMES[split][LABELS.index(label)])
                text = frame.format(skill=skill, action=rng.choice(SKILLS[skill]))
                rows.append(dict(split=split, cv_id=f'cv-{pair_id}',
                                 posting_id=f'job-{pair_id}', industry=industry,
                                 role='Data Analyst', requirement_text=skill,
                                 cv_excerpt=text, label=label))
    return pd.DataFrame(rows)


REQUIRED = ['split', 'cv_id', 'posting_id', 'requirement_text', 'cv_excerpt', 'label']


def clean(data):
    if set(REQUIRED).difference(data):
        raise ValueError('Required columns: ' + ', '.join(REQUIRED))
    data = data.apply(lambda column: column.str.strip()).replace('', pd.NA)
    missing = data.isna().sum().astype(int)
    if not data.split.dropna().isin(SPLITS).all():
        raise ValueError('Unexpected split name.')
    unique = data.drop_duplicates()
    complete = unique.dropna(subset=REQUIRED)
    training = complete.loc[complete.split.eq('train')]
    optional_missing = training.drop(columns=REQUIRED).isna().mean()
    dropped = optional_missing[optional_missing.gt(0.90)].index.tolist()
    return complete.drop(columns=dropped), {
        'input_rows': len(data),
        'missing_by_column': missing.to_dict(),
        'exact_duplicate_rows_removed': len(data) - len(unique),
        'rows_missing_required_values_removed': len(unique) - len(complete),
        'optional_columns_dropped': dropped,
    }


def validate(data):
    if set(data.split) != set(SPLITS) or set(data.label) != set(LABELS):
        raise ValueError('Expected all three split names and evidence labels.')
    for key in ('cv_id', 'posting_id'):
        if data.groupby(key).split.nunique().gt(1).any():
            raise ValueError(f'{key} appears in more than one split.')
    if data.duplicated(['cv_id', 'posting_id', 'requirement_text']).any():
        raise ValueError('Expected one label per CV, posting and requirement.')
    for split in SPLITS:
        if set(data.loc[data.split.eq(split), 'label']) != set(LABELS):
            raise ValueError(f'{split} must contain all three labels.')


def baseline(data):
    def normalize(text):
        return ' '.join(re.findall(r'\w+', text.casefold()))
    return [
        'evidence found' if f' {normalize(skill)} ' in f' {normalize(text)} '
        else 'no evidence found'
        for skill, text in zip(data.requirement_text, data.cv_excerpt)
    ]


FEATURES = ['requirement_text', 'cv_excerpt']


def features(data):
    return data[FEATURES]


def build_pipeline():
    preprocessor = ColumnTransformer([
        ('requirement', TfidfVectorizer(ngram_range=(1, 2)), 'requirement_text'),
        ('cv', TfidfVectorizer(ngram_range=(1, 2)), 'cv_excerpt'),
    ])
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000)),
    ])


def select_model(X_train, y_train, X_val, y_val):
    base = build_pipeline()
    families = [
        ('Logistic regression', LogisticRegression(max_iter=1000), 'C', (0.1, 1.0, 10.0)),
        ('Linear SVM', LinearSVC(random_state=42), 'C', (0.1, 1.0, 10.0)),
        ('Complement naive Bayes', ComplementNB(), 'alpha', (0.1, 1.0, 10.0)),
    ]
    candidates = []
    for name, classifier, parameter, values in families:
        for value in values:
            model = clone(base).set_params(
                classifier=clone(classifier), **{f'classifier__{parameter}': value}
            )
            model.fit(X_train, y_train)
            metrics = evaluate(y_val, model.predict(X_val))
            candidates.append((f'{name} ({parameter}={value})', model, metrics))
    # max preserves the first candidate on an exact tie.
    best_name, best_model, _ = max(candidates, key=lambda item: item[2]['macro_f1'])
    return best_name, best_model, {name: metrics for name, _, metrics in candidates}


def evaluate(expected, predicted):
    report = classification_report(expected, predicted, labels=LABELS,
                                   output_dict=True, zero_division=0)
    return {
        'macro_f1': report['macro avg']['f1-score'],
        'per_label': {label: report[label] for label in LABELS},
        'confusion_matrix': confusion_matrix(expected, predicted, labels=LABELS).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Labeled CSV with preassigned splits')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--save-model', action='store_true',
                        help='Save the selected pipeline alongside the evaluation outputs')
    args = parser.parse_args()
    data = (pd.read_csv(args.input, dtype=str, keep_default_na=False)
            if args.input else make_demo())
    data, cleaning = clean(data)
    validate(data)
    train, val, test = [data.loc[data.split.eq(name)] for name in SPLITS]

    best_name, model, validation_results = select_model(
        features(train), train.label, features(val), val.label
    )
    majority_label = train.label.mode().iloc[0]
    prediction = model.predict(features(test))
    baseline_prediction = baseline(test)
    results = {
        'cleaning': cleaning,
        'data_source': 'provided labeled CSV' if args.input else 'synthetic examples',
        'split_counts': {name: {'rows': len(part), 'cvs': part.cv_id.nunique(),
                                 'postings': part.posting_id.nunique(),
                                 'labels': part.label.value_counts().to_dict()}
                         for name, part in zip(SPLITS, [train, val, test])},
        'label_order': LABELS,
        'validation_keyword_baseline': evaluate(val.label, baseline(val)),
        'validation_models': validation_results,
        'selected_model': best_name,
        'majority_label_from_train': majority_label,
        'test_majority_baseline': evaluate(test.label, [majority_label] * len(test)),
        'test_keyword_baseline': evaluate(test.label, baseline_prediction),
        'test_model': evaluate(test.label, prediction),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, part in zip(SPLITS, [train, val, test]):
        part.to_csv(args.output / f'{name}.csv', index=False)
    test.assign(model_prediction=prediction, keyword_prediction=baseline_prediction,
                majority_prediction=majority_label).to_csv(
        args.output / 'test_predictions.csv', index=False)
    (args.output / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    if args.save_model:
        joblib.dump(model, args.output / 'cv_evidence.joblib', compress=3)
    print(f'Selected {best_name}, test macro F1: '
          f'{results["test_model"]["macro_f1"]:.3f} model, '
          f'{results["test_keyword_baseline"]["macro_f1"]:.3f} keyword baseline')


if __name__ == '__main__':
    main()
