"""PhishGuard: ML-based phishing URL detection (educational prototype).

Uses character n-grams plus lexical URL features. URLs are not opened or sent
anywhere. Dataset labels: 0=phishing, 1=legitimate (UCI PhiUSIIL).
"""
from __future__ import annotations
import argparse, ipaddress, json, sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler

ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT/'data'; MODEL_DIR=ROOT/'models'; REPORT_DIR=ROOT/'reports'
MODEL_PATH=MODEL_DIR/'phishguard_model.joblib'; METRICS_PATH=REPORT_DIR/'evaluation_metrics.json'
RANDOM_STATE=42
FEATURE_NAMES=['url_length','hostname_length','path_length','query_length','count_dots','count_hyphens','count_underscores','count_at','count_question_marks','count_equal','count_ampersands','count_percent','count_digits','digit_ratio','count_subdomains','uses_https','uses_http','has_ip_hostname','has_punycode','has_suspicious_word','has_shortener_hint','hostname_is_empty','count_slashes','count_colons']
SUSPICIOUS_WORDS=('login','verify','account','update','secure','signin','banking','password','confirm','wallet','payment','support','recover')
SHORTENER_HINTS=('bit.ly','tinyurl.com','t.co','is.gd','cutt.ly','ow.ly')

def extract_url_features(raw_url: str) -> dict[str,float]:
    value=str(raw_url).strip()
    candidate=value if '://' in value else 'http://'+value
    try:
        parts=urlsplit(candidate); hostname=(parts.hostname or '').lower().rstrip('.')
        path=parts.path or ''; query=parts.query or ''
    except ValueError:
        parts=urlsplit(''); hostname=path=query=''
    n=max(len(value),1); digits=sum(ch.isdigit() for ch in value)
    try: ipaddress.ip_address(hostname); has_ip=1
    except ValueError: has_ip=0
    labels=[x for x in hostname.split('.') if x]; low=value.lower(); counts=Counter(value)
    return {'url_length':len(value),'hostname_length':len(hostname),'path_length':len(path),'query_length':len(query),
      'count_dots':counts['.'],'count_hyphens':counts['-'],'count_underscores':counts['_'],'count_at':counts['@'],
      'count_question_marks':counts['?'],'count_equal':counts['='],'count_ampersands':counts['&'],'count_percent':counts['%'],
      'count_digits':digits,'digit_ratio':digits/n,'count_subdomains':max(len(labels)-2,0),
      'uses_https':int(parts.scheme.lower()=='https'),'uses_http':int(parts.scheme.lower()=='http'),'has_ip_hostname':has_ip,
      'has_punycode':int('xn--' in hostname),'has_suspicious_word':int(any(w in low for w in SUSPICIOUS_WORDS)),
      'has_shortener_hint':int(any(h in hostname for h in SHORTENER_HINTS)),'hostname_is_empty':int(not hostname),
      'count_slashes':counts['/'],'count_colons':counts[':']}

def lexical_dicts(urls): return [extract_url_features(u) for u in urls]
def find_column(columns,candidates):
    lookup={str(c).strip().lower():c for c in columns}
    for x in candidates:
        if x.lower() in lookup:return lookup[x.lower()]
    return None

def load_dataset(csv_path:Path):
    if not csv_path.exists(): raise FileNotFoundError(f'Dataset not found: {csv_path}. Run --download-data or place the UCI CSV in data/.')
    df=pd.read_csv(csv_path,low_memory=False)
    uc=find_column(df.columns,['URL','uri','website']); lc=find_column(df.columns,['label','target','class','result'])
    if uc is None or lc is None: raise ValueError('Could not find URL and label columns. Found: '+', '.join(map(str,df.columns)))
    out=df[[uc,lc]].copy(); out.columns=['url','label']; out['url']=out.url.astype('string').str.strip()
    out=out.dropna(subset=['url','label']); out=out[out.url.str.len()>0].drop_duplicates(subset=['url'],keep='first')
    out['label']=pd.to_numeric(out.label,errors='coerce'); out=out.dropna(subset=['label'])
    labels=set(out.label.unique())
    if labels!={0,1}: raise ValueError(f'Expected labels 0 and 1, found {sorted(labels)}. Verify label mapping.')
    out['label']=out.label.astype(int)
    return out.reset_index(drop=True)

def make_model():
    # Character n-grams are already TF-IDF normalized. Scale the numeric lexical
    # branch too; otherwise raw URL counts/lengths can dominate the sparse n-grams.
    lexical_features = Pipeline([
      ('extract',FunctionTransformer(lexical_dicts,validate=False)),
      ('dict',DictVectorizer()),
      ('scale',MaxAbsScaler()),
    ])
    feats=FeatureUnion([
      ('character_ngrams',TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=150000,sublinear_tf=True,dtype=np.float32)),
      ('lexical',lexical_features),
    ])
    # The dataset is moderately imbalanced, not extremely so; do not up-weight
    # phishing by default because that can increase false positives on new domains.
    return Pipeline([('features',feats),('classifier',SGDClassifier(loss='log_loss',alpha=1e-5,class_weight=None,max_iter=1000,tol=1e-3,random_state=RANDOM_STATE))])

def download_dataset():
    try: from ucimlrepo import fetch_ucirepo
    except ImportError as e: raise RuntimeError('Install dependencies first: pip install -r requirements.txt') from e
    DATA_DIR.mkdir(parents=True,exist_ok=True)
    print('Downloading UCI dataset 967; internet access is required...')
    ds=fetch_ucirepo(id=967); df=pd.concat([ds.data.features,ds.data.targets],axis=1)
    p=DATA_DIR/'PhiUSIIL_Phishing_URL_Dataset.csv'; df.to_csv(p,index=False); print(f'Saved dataset: {p}'); return p

def evaluate(y_true,pred,proba,classes,name):
    out={'model':name,'accuracy':float(accuracy_score(y_true,pred)),
      'precision_phishing_label_0':float(precision_score(y_true,pred,pos_label=0,zero_division=0)),
      'recall_phishing_label_0':float(recall_score(y_true,pred,pos_label=0,zero_division=0)),
      'f1_phishing_label_0':float(f1_score(y_true,pred,pos_label=0,zero_division=0)),
      'confusion_matrix_labels_order_0_then_1':confusion_matrix(y_true,pred,labels=[0,1]).tolist(),
      'classification_report':classification_report(y_true,pred,labels=[0,1],target_names=['phishing (0)','legitimate (1)'],output_dict=True,zero_division=0)}
    if proba is not None and 1 in list(classes): out['roc_auc_legitimate_class_1']=float(roc_auc_score(y_true,proba[:,list(classes).index(1)]))
    return out

def train_and_evaluate(csv_path):
    df=load_dataset(Path(csv_path)); print(f'Unique labeled URLs after cleaning: {len(df):,}')
    print('Labels (0=phishing, 1=legitimate):\n'+df.label.value_counts().sort_index().to_string())
    X=df.url.astype(str); y=df.label
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=RANDOM_STATE,stratify=y)
    baseline=DummyClassifier(strategy='most_frequent').fit(np.zeros((len(Xtr),1)),ytr)
    bp=baseline.predict(np.zeros((len(Xte),1)))
    baseline_metrics=evaluate(yte,bp,None,baseline.classes_,'Most-frequent baseline')
    model=make_model(); print('Training character n-gram + lexical-feature classifier...'); model.fit(Xtr,ytr)
    pred=model.predict(Xte); proba=model.predict_proba(Xte); metrics_model=evaluate(yte,pred,proba,model.classes_,'Character n-gram + lexical SGD classifier')
    MODEL_DIR.mkdir(exist_ok=True); REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump({'model':model,'label_mapping':{0:'phishing',1:'legitimate'},'dataset':'UCI PhiUSIIL Phishing URL Dataset (ID 967)','features':'character TF-IDF n-grams plus MaxAbs-scaled URL lexical features','model_version':'scaled_lexical_v2','random_state':RANDOM_STATE},MODEL_PATH)
    metrics={'dataset_rows_after_cleaning':int(len(df)),'train_rows':int(len(Xtr)),'test_rows':int(len(Xte)),
      'label_mapping':{'0':'phishing','1':'legitimate'},'split':'80/20 stratified random URL-level split; random_state=42',
      'limitations':['URLs from the same domain can occur in both partitions.','Random split scores may overstate performance on unseen domains or future URLs.','The model inspects URL text only; it does not fetch pages, redirects, certificates, or reputation data.','Probability scores are model estimates, not calibrated guarantees.','A false positive or false negative is possible.'],
      'baseline':baseline_metrics,'model':metrics_model}
    METRICS_PATH.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    for m in (baseline_metrics,metrics_model):
      print(f"\n{m['model']}: accuracy={m['accuracy']:.4f}, phishing precision={m['precision_phishing_label_0']:.4f}, phishing recall={m['recall_phishing_label_0']:.4f}, phishing F1={m['f1_phishing_label_0']:.4f}")
      print('Confusion matrix [true rows / predicted columns, labels 0 then 1]:',m['confusion_matrix_labels_order_0_then_1'])
    print(f'\nSaved model: {MODEL_PATH}\nSaved metrics: {METRICS_PATH}')
    print('Important: these random-split metrics do not establish real-world reliability.')
    return metrics

def predict_url(url):
    if not url or not url.strip(): raise ValueError('Enter a non-empty URL.')
    if not MODEL_PATH.exists(): raise FileNotFoundError('Trained model missing. Run --train first.')
    cleaned=url.strip()
    bundle=joblib.load(MODEL_PATH); model=bundle['model']; pred=int(model.predict([cleaned])[0])
    print(f'Input: {cleaned}\nPrediction: {bundle["label_mapping"][pred]} (label {pred})')
    if hasattr(model,'predict_proba'):
      p=model.predict_proba([cleaned])[0]; cls=list(model.classes_)
      if pred in cls: print(f'Uncalibrated model score for predicted class: {p[cls.index(pred)]:.1%}')
    print('Caution: this is a URL-text estimate, not a safety verdict. The score is not a calibrated probability; false positives and false negatives are possible.')
    print('The URL was not opened or transmitted.')

def main():
    ap=argparse.ArgumentParser(description='PhishGuard phishing URL classifier')
    ap.add_argument('--download-data',action='store_true'); ap.add_argument('--train',action='store_true')
    ap.add_argument('--csv',type=Path,default=DATA_DIR/'PhiUSIIL_Phishing_URL_Dataset.csv'); ap.add_argument('--predict')
    args=ap.parse_args()
    try:
      csv=args.csv
      if args.download_data: csv=download_dataset()
      if args.train: train_and_evaluate(csv)
      if args.predict is not None: predict_url(args.predict)
      if not (args.download_data or args.train or args.predict is not None): ap.print_help()
    except Exception as e: print(f'ERROR: {e}',file=sys.stderr); sys.exit(1)
if __name__=='__main__': main()
