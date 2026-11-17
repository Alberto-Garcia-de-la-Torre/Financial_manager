"""Day 41: the model interface and the registry behind `finmgr train --model`.

The acceptance criterion: `finmgr train --model ridge` and `--model lgbm` both
resolve through the registry. Beyond that, these tests pin the contract every
later model relies on — both registered models satisfy the protocol, learn an
obvious signal, refuse to predict before fitting or on a reshuffled feature
set, reject misspelt parameters, and are deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from finmgr import models
from finmgr.cli import app
from finmgr.config import SETTINGS_PATH_ENV
from finmgr.models import registry

NAMES = ["lgbm", "ridge"]


def make_data(n: int = 2000, seed: int = 41) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["mom", "vol", "noise"])
    y = pd.Series(2.0 * X["mom"] - 1.0 * X["vol"] + 0.1 * rng.standard_normal(n))
    return X, y


def test_both_models_are_registered():
    # Day 44 added the classifiers; the regression contract below covers NAMES only.
    assert models.available() == sorted([*NAMES, "lgbm_clf", "logit"])


@pytest.mark.parametrize("name", NAMES)
def test_create_returns_a_fresh_model_honouring_the_protocol(name):
    model = models.create(name)
    assert isinstance(model, models.Model)
    assert model.name == name
    assert model.params == type(model).defaults
    assert models.create(name) is not model


@pytest.mark.parametrize("name", NAMES)
def test_fit_then_predict_learns_an_obvious_signal(name):
    X, y = make_data()
    train, test = slice(0, 1500), slice(1500, None)
    model = models.create(name).fit(X[train], y[train])
    pred = model.predict(X[test])
    assert pred.shape == (500,)
    assert pred.dtype == float
    assert np.corrcoef(pred, y[test])[0, 1] > 0.9


@pytest.mark.parametrize("name", NAMES)
def test_refits_are_deterministic(name):
    X, y = make_data()
    a = models.create(name).fit(X, y).predict(X)
    b = models.create(name).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", NAMES)
def test_predict_before_fit_is_an_error(name):
    X, _ = make_data(10)
    with pytest.raises(models.NotFittedError):
        models.create(name).predict(X)


@pytest.mark.parametrize("name", NAMES)
def test_predict_on_a_different_feature_set_is_an_error(name):
    X, y = make_data()
    model = models.create(name).fit(X, y)
    with pytest.raises(ValueError, match="differ from the fitted"):
        model.predict(X[["vol", "mom", "noise"]])  # same columns, other order
    with pytest.raises(ValueError, match="differ from the fitted"):
        model.predict(X[["mom", "vol"]])


def test_mismatched_lengths_are_an_error():
    X, y = make_data(100)
    with pytest.raises(ValueError, match="rows"):
        models.create("ridge").fit(X, y.iloc[:-1])


def test_overrides_replace_defaults_and_typos_fail():
    model = models.create("ridge", alpha=10.0)
    assert model.params == {"alpha": 10.0, "fit_intercept": True}
    model.params["alpha"] = 0.0  # a copy: mutating it changes nothing
    assert model.params["alpha"] == 10.0
    with pytest.raises(ValueError, match="unknown parameter"):
        models.create("ridge", alpah=10.0)


def test_unknown_model_names_the_choices():
    with pytest.raises(models.UnknownModel, match="choose from lgbm, lgbm_clf, logit, ridge"):
        models.create("xgb")


def test_a_name_cannot_be_registered_twice_to_different_classes():
    class Other(models.EstimatorModel):
        def _estimator(self):  # pragma: no cover - never built
            raise NotImplementedError

    with pytest.raises(ValueError, match="already registered"):
        registry.register("ridge")(Other)
    assert models.resolve("ridge").__name__ == "RidgeModel"


# --- `finmgr train --model` ------------------------------------------------------------


@pytest.fixture
def cli(tmp_path, monkeypatch, write_settings):
    settings = write_settings(
        {"data_dir": str(tmp_path / "data"), "log_dir": str(tmp_path / "logs")}
    )
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings))
    return CliRunner()


@pytest.mark.parametrize("name", NAMES)
def test_train_resolves_both_models_through_the_registry(cli, monkeypatch, name):
    resolved: list[str] = []
    real = registry.create

    def spy(model_name, **params):
        resolved.append(model_name)
        return real(model_name, **params)

    monkeypatch.setattr(registry, "create", spy)
    monkeypatch.setattr(models, "create", spy)
    result = cli.invoke(app, ["train", "--model", name, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert resolved == [name]
    assert f"model '{name}'" in result.output


def test_train_passes_typed_param_overrides(cli):
    result = cli.invoke(
        app,
        ["train", "-m", "lgbm", "-p", "n_estimators=50", "-p", "learning_rate=0.1", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "50" in result.output and "0.1" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--model", "xgb"], "unknown model 'xgb'"),
        (["--model", "ridge", "-p", "alpah=1"], "unknown parameter"),
        (["--model", "ridge", "-p", "alpha"], "KEY=VALUE"),
    ],
)
def test_train_rejects_bad_input_as_usage_errors(cli, args, message):
    result = cli.invoke(app, ["train", *args, "--dry-run"])
    assert result.exit_code == 2
    assert message in " ".join(result.output.split())


def test_train_lists_the_registry(cli):
    result = cli.invoke(app, ["train", "--list"])
    assert result.exit_code == 0, result.output
    assert "RidgeModel" in result.output and "LGBMModel" in result.output


def test_train_without_a_feature_panel_fails_cleanly(cli, monkeypatch):
    # Day 42 made `train` real (tests/test_walkforward.py drives it end to end);
    # with no panel in the store it must fail with a message, not a traceback.
    from finmgr.models import walkforward

    def no_panel(**_):
        raise FileNotFoundError("no feature panel; run `finmgr features --full`")

    monkeypatch.setattr(walkforward, "load_dataset", no_panel)
    result = cli.invoke(app, ["train", "--model", "ridge"])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
