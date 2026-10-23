import importlib.util
from pathlib import Path
from sklearn.preprocessing import MaxAbsScaler

SCRIPT = Path(__file__).resolve().parents[1] / 'AnoushkaJamalapur_PhishGuard.py'
spec = importlib.util.spec_from_file_location('phishguard_scaled', SCRIPT)
app = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(app)


def test_lexical_feature_branch_is_scaled():
    model = app.make_model()
    feature_union = model.named_steps['features']
    lexical = dict(feature_union.transformer_list)['lexical']
    assert isinstance(lexical.named_steps['scale'], MaxAbsScaler)


def test_predictions_do_not_bypass_model_with_demo_allowlist():
    assert not hasattr(app, 'trusted_domain_match')
