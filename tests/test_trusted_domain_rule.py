import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'AnoushkaJamalapur_PhishGuard.py'
spec = importlib.util.spec_from_file_location('phishguard_no_allowlist', SCRIPT)
app = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(app)


def test_all_urls_use_the_ml_prediction_path():
    # The previous demo allowlist masked model failures on selected examples.
    assert not hasattr(app, 'trusted_domain_match')
