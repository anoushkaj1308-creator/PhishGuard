import importlib.util
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "AnoushkaJamalapur_PhishGuard.py"
spec = importlib.util.spec_from_file_location("phishguard", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_feature_schema_and_https():
    f = module.extract_url_features("https://example.com/a?x=1")
    assert list(f.keys()) == module.FEATURE_NAMES
    assert f["uses_https"] == 1
    assert f["hostname_length"] == len("example.com")

def test_ip_hostname_detection():
    assert module.extract_url_features("http://192.0.2.10/login")["has_ip_hostname"] == 1

def test_suspicious_lexical_features():
    f = module.extract_url_features("https://example.com/login?next=%2F")
    assert f["has_suspicious_word"] == 1
    assert f["count_percent"] >= 1

def test_dataset_has_expected_columns_and_labels():
    df = module.load_dataset(Path(__file__).resolve().parents[1] / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv")
    assert len(df) > 200_000
    assert set(df["label"].unique()) == {0, 1}
    assert set(df.columns) == {"url", "label"}
