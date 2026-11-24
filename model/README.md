# Résumé Studio model

Résumé Studio helps applicants review whether their CV shows evidence for the skills in a job posting. The browser app checks keyword coverage and lets you review the strength of each match. It lists gaps in skills marked essential first. The Python model in this folder is a separate research prototype that classifies a job requirement and one CV excerpt. [Open the browser app](https://witchayada-t.github.io/Digital-Talent/)

## Try the model

Open a terminal in this `model` folder. The release was tested with Python 3.12 and the versions in [requirements.txt](requirements.txt).

```bash
python -m pip install -r requirements.txt
python predict.py --requirement "SQL" --excerpt "I independently joined database tables."
```

The command uses the included trained pipeline and prints one label.

| Label | Meaning |
| --- | --- |
| `evidence found` | The model reads the excerpt as evidence supporting the requirement. |
| `no evidence found` | The model reads the excerpt as providing no evidence for the requirement. |
| `unclear` | The model reads the excerpt as ambiguous or incomplete. Review the wording and context. |

Check the original excerpt before acting on a label. The model was trained and tested on synthetic examples and has no independently verified performance on real CVs. It evaluates a single requirement at a time.

## Training and results

Separate TF-IDF transformers encode the requirement and the excerpt. The benchmark fits logistic regression, linear SVM and Complement Naive Bayes on **360 training rows** and selects the setting by macro F1 on **120 validation rows**. Complement Naive Bayes with `alpha=0.1` won the validation comparison. The training split contains 120 examples of each label. CV IDs, posting IDs and normalised CV excerpts are disjoint across train, validation and test.

| Method | Macro F1 on 120 synthetic test rows |
| --- | ---: |
| Complement Naive Bayes | **0.649** |
| Literal keyword baseline | 0.289 |
| Majority label baseline | 0.173 |

The model identified **14 of 39** unclear examples and treated **36 of 78** absent or unclear examples as evidence found. The [full metrics](validation/demo/metrics.json) include per-label precision, recall and the confusion matrix.

Reproduce the benchmark and save a newly trained model in a separate results folder.

```bash
python validation/run.py --output results --save-model
```

The [CV evidence notebook](validation/CV_Evidence_Pipeline.ipynb) walks through the data, label balance, leakage checks and evaluation.

## Occupation skill groups

The [K-means notebook](notebooks/Occupation_Clustering_K4.ipynb) profiles 910 occupations using 14 standardised O*NET skill Importance scores. It uses **K = 4** to show the resource management and team coordination profile separately. The four groups contain 128, 214, 235 and 333 occupations. The occupation groups are descriptive. The evidence classifier uses only the requirement and excerpt texts.

- [Occupation and cluster assignments](data/occupation_cluster_assignments.csv)
- [Skill rankings by cluster](data/cluster_skill_rankings.csv)
- [Skill Importance and Level notebook](notebooks/Skill_Importance_and_Level_EDA.ipynb)

The notebook inputs are transferable-skill exports from O*NET OnLine for the U.S. Department of Labor. Résumé Studio reshapes the exports and adds cluster assignments and summaries. [O*NET OnLine](https://www.onetonline.org/) · [O*NET OnLine licence](https://www.onetonline.org/help/license)
