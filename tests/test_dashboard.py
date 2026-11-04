from pathlib import Path

from streamlit.testing.v1 import AppTest

DASHBOARD = str(Path(__file__).resolve().parent.parent / "src" / "dashboard.py")


def test_dashboard_runs_and_scores_a_transaction():
    app = AppTest.from_file(DASHBOARD, default_timeout=60).run()

    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert "Fraud probability" in labels
    assert "Fraud caught" in labels


def test_dashboard_threshold_slider_changes_decisions():
    app = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    caught_default = next(m.value for m in app.metric if m.label == "Fraud caught")

    app.slider[0].set_value(0.99).run()
    caught_strict = next(m.value for m in app.metric if m.label == "Fraud caught")

    assert not app.exception
    assert caught_strict != caught_default
