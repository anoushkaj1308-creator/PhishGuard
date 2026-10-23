# PhishGuard: Machine-Learning-Based Phishing URL Detection

## Project overview

PhishGuard is a defensive cybersecurity project that uses a
machine-learning model to classify a URL string as **phishing (label
0)** or **legitimate (label 1)**. It extracts lexical features from the
URL itself. The program does not open the website or submit the entered
URL to a third-party checking service.

**Safety note:** A prediction is an estimate, not proof that a website
is safe or malicious. Do not use the result as the only basis for a
security decision.

## Dataset

-   **Dataset:** PhiUSIIL Phishing URL (Website) Dataset
-   **Source:** UCI Machine Learning Repository
-   **Official page:**
    https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
-   **Labels used:** `0 = phishing`; `1 = legitimate`
-   **Preprocessing:** Empty URLs and duplicate URL strings are removed
    before the URL-level train/test split. The project derives features
    from the URL string so that the same feature extraction can be
    applied to a URL entered for prediction.

Keep the dataset in the project's `data` folder in the location/name
expected by the supplied script. The dataset is large, so include it in
a submission only if the submission instructions request it or permit
it.

## Requirements

-   Python 3
-   Packages listed in `requirements.txt`
-   The project's dataset file for training
-   Internet access may be needed to install packages or obtain the
    dataset

## Setup on Windows

Open Command Prompt in the project folder (the folder containing
`AnoushkaJamalapur_PhishGuard.py`).

Create a virtual environment:

``` bat
py -m venv .venv
```

Install the required packages:

``` bat
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you have already created `.venv` and installed the requirements, you
do not need to repeat these steps.

## Train and evaluate the model

Run:

``` bat
.venv\Scripts\python.exe AnoushkaJamalapur_PhishGuard.py --train
```

The script trains the revised classifier, evaluates it on a held-out
stratified random split, and saves the trained model and evaluation
metrics. The expected output files are:

-   `models\phishguard_model.joblib`
-   `reports\evaluation_metrics.json`

Use the metrics produced by the current script and current dataset. Do
not substitute results from an earlier baseline run.

## Predict a URL

After training, run:

``` bat
.venv\Scripts\python.exe AnoushkaJamalapur_PhishGuard.py --predict "https://www.wikipedia.org/"
```

Replace the example with a URL you are permitted to analyze. The script
evaluates the URL text; it does not visit the website. Avoid entering
URLs containing private session tokens or personal information.

## Model evaluation

The revised model's recorded evaluation on the current random URL-level
test split was:

  Metric                           Result
  ------------------------------ --------
  Accuracy                         99.54%
  Phishing precision (label 0)     99.98%
  Phishing recall (label 0)        98.93%
  Phishing F1-score (label 0)      99.45%
  ROC-AUC (legitimate class)       0.9987

Confusion matrix, with rows as true labels and columns as predicted
labels in the order `[0 = phishing, 1 = legitimate]`:

``` text
[[19889,   215],
 [    3, 26967]]
```

The values above describe the recorded evaluation run; results may
change if the dataset, preprocessing, dependencies, or random split
changes.

## Features and approach

The classifier combines character n-gram features from the URL with
lexical URL features. These can capture spelling patterns, URL
structure, symbol counts, and other text-based indicators that may
differ between phishing and legitimate URLs.

## Limitations

-   The evaluation uses a **random URL-level split**, not a
    domain-disjoint or time-based split. Similar URLs or URLs from the
    same domain may occur in both training and test partitions, so the
    reported score may overstate performance on entirely unseen domains.
-   The classifier analyzes URL text only. It does not inspect webpage
    content, redirects, domain reputation, TLS certificates, or live
    threat-intelligence feeds.
-   Model scores are not calibrated probabilities and should not be
    interpreted as a guarantee of risk.
-   Legitimate URLs can contain suspicious-looking patterns, and
    phishing URLs can appear ordinary.
-   This is an educational prototype, not a replacement for browser
    protections, security tools, or human review.

## Tests

Run the project's tests from the project folder:

``` bat
.venv\Scripts\python.exe -m pytest tests -v
```

A passing test suite checks the implemented behavior; it does not prove
that the classifier can identify every phishing URL.

## Project files

``` text
PhishGuard_Submission/
├── AnoushkaJamalapur_PhishGuard.py
├── AnoushkaJamalapur_ProjectReport.docx
├── README.md
├── requirements.txt
├── SUBMISSION_CHECKLIST.txt
├── data/
├── tests/
├── models/                  # created by training
└── reports/
    └── evaluation_metrics.json  # created by training
```

## Dataset citation

Prasad, A., & Chandra, S. (2024). *PhiUSIIL: A diverse security profile
empowered phishing URL detection framework based on similarity index and
incremental learning*. Computers & Security. Dataset record:
https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

The dataset is provided through the UCI Machine Learning Repository.
Review the dataset page for attribution and licensing terms before
redistributing the dataset.
