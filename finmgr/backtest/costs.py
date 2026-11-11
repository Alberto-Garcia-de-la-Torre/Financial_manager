"""Costs and frictions: what a fill really costs, and the prices fills happen at.

    from finmgr.backtest.costs import CostModel, read_backtest_prices
    from finmgr.backtest.engine import run_backtest

    prices = read_backtest_prices(as_of="2026-10-15", start="2020-01-01")
    costs = CostModel.from_settings()                 # config/settings.yaml, `costs:`
    result = run_backtest(scores, prices, top_k=10, hold=21, costs=costs)
    result = run_backtest(scores, prices, top_k=10, hold=21, costs=CostModel.flat(50))

**The model.** Every fill pays a one-way cost, in basis points of the traded
notional, made of three parts:

    one_way_bps = commission_bps + (spread_bps / 2 + slippage_bps) * scale

* `commission_bps` — the broker's fee, a flat share of the notional.
* `spread_bps` — the quoted bid-ask spread. A round trip crosses all of it, so
  each side pays half.
* `slippage_bps` — market impact and the open auction drifting away from the
  price the order was sized at.
* `scale` — 1 unless `liquidity_scaling` is on. Then it is
  `liquidity_ref_eur / adv`, clipped to `[min_liquidity_scale,
  max_liquidity_scale]`, with `adv` the name's 21-session average daily euro
  volume (day 29's `adv_eur_21`) on the decision date: a name trading half the
  reference volume pays twice the spread and slippage, and a name of unknown
  liquidity pays the maximum. The commission is not scaled — a broker does not
  charge more for a thin stock.

A buy of budget B spends exactly B: `B / (1 + c)` goes into shares at the fill
price and `B * c / (1 + c)` is the cost. A sale of notional V returns
`V * (1 - c)`. Costs are reported per fill (`trades.cost_eur`) and per day
(`equity.cost_eur`); the equity curve is net of them.

**Every number is configurable**: as arguments here, or in the `costs:`
section of `config/settings.yaml` (:meth:`CostModel.from_settings`).
:data:`NO_COSTS` is the frictionless model, kept for identities and tests;
:meth:`CostModel.flat` puts a whole one-way cost in one number.

**Fill prices.** The engine never trades at the close the score was computed
from: an order decided at t's close fills at the *next* session's open
(`fill="next_open"`, the default). :func:`read_backtest_prices` supplies that
open, in EUR, back-adjusted the same way as the close, so a fill and a mark of
the same session are on the same scale.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from finmgr.data import pit

if TYPE_CHECKING:
    from finmgr.config import Settings

#: The EUR open, back-adjusted like `adj_close_eur`.
OPEN_COLUMN = "adj_open_eur"
#: The liquidity column the cost scale reads (day 29's average daily euro volume).
LIQUIDITY_COLUMN = "adv_eur_21"
#: What :func:`read_backtest_prices` returns.
BACKTEST_PRICE_COLUMNS: tuple[str, ...] = (
    "ticker",
    "date",
    "adj_close_eur",
    OPEN_COLUMN,
    LIQUIDITY_COLUMN,
)

BPS = 1e-4


class CostModel(BaseModel):
    """Commission, spread and slippage in basis points, optionally scaled by inverse liquidity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    commission_bps: float = Field(default=5.0, ge=0, description="Broker fee, per side.")
    spread_bps: float = Field(
        default=10.0, ge=0, description="Full quoted bid-ask spread; each side pays half."
    )
    slippage_bps: float = Field(default=5.0, ge=0, description="Impact and drift, per side.")
    liquidity_scaling: bool = Field(
        default=True, description="Scale spread and slippage by liquidity_ref_eur / adv."
    )
    liquidity_ref_eur: float = Field(
        default=100e6, gt=0, description="Average daily euro volume at which scale is 1."
    )
    min_liquidity_scale: float = Field(
        default=1.0, ge=0, description="Floor on the scale: very liquid names pay at least this."
    )
    max_liquidity_scale: float = Field(
        default=10.0, ge=0, description="Cap on the scale; also what unknown liquidity pays."
    )

    @model_validator(mode="after")
    def _ordered_scales(self) -> CostModel:
        if self.min_liquidity_scale > self.max_liquidity_scale:
            raise ValueError(
                f"min_liquidity_scale {self.min_liquidity_scale} is above "
                f"max_liquidity_scale {self.max_liquidity_scale}"
            )
        return self

    @classmethod
    def flat(cls, bps: float) -> CostModel:
        """A one-way cost of `bps` on every fill, whatever the name's liquidity."""
        return cls(commission_bps=bps, spread_bps=0.0, slippage_bps=0.0, liquidity_scaling=False)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> CostModel:
        """The `costs:` section of the project settings."""
        from finmgr.config import load_settings

        return (settings or load_settings()).costs

    def scale(self, adv: np.ndarray | float) -> np.ndarray:
        """The spread-and-slippage multiplier for each average daily euro volume."""
        adv = np.asarray(adv, dtype=float)
        if not self.liquidity_scaling:
            return np.ones_like(adv)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = self.liquidity_ref_eur / adv
        raw = np.where(np.isfinite(raw) & (adv > 0), raw, self.max_liquidity_scale)
        return np.clip(raw, self.min_liquidity_scale, self.max_liquidity_scale)

    def one_way_bps(self, adv: np.ndarray | float = np.nan) -> np.ndarray:
        """The one-way cost of a fill, in basis points, for each average daily euro volume."""
        variable = (self.spread_bps / 2 + self.slippage_bps) * self.scale(adv)
        return self.commission_bps + variable

    def rate(self, adv: np.ndarray | float = np.nan) -> np.ndarray:
        """:meth:`one_way_bps` as a fraction of the notional."""
        return self.one_way_bps(adv) * BPS

    @property
    def needs_liquidity(self) -> bool:
        """Whether this model reads a liquidity column at all."""
        return self.liquidity_scaling and (self.spread_bps > 0 or self.slippage_bps > 0)


#: The frictionless model: free fills. For identities and tests, never for a verdict.
NO_COSTS = CostModel(commission_bps=0.0, spread_bps=0.0, slippage_bps=0.0, liquidity_scaling=False)


def adjusted_open(frame: pd.DataFrame) -> pd.Series:
    """`open * adj_close_eur / close`: the open on the scale of the EUR total-return close.

    The ratio carries both the back-adjustment factor and the session's EUR
    rate, which apply to the whole session, open included.
    """
    return frame["open"] * frame["adj_close_eur"] / frame["close"]


def read_backtest_prices(
    tickers: Iterable[str] | str | None = None,
    *,
    as_of: pit.AsOf,
    start: date | str | None = None,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """What the engine needs from the store: EUR close and open, and liquidity.

    Returns :data:`BACKTEST_PRICE_COLUMNS`, sorted by `(ticker, date)`, one row
    per stored session. Nothing after `as_of` is read.
    """
    from finmgr.config import load_settings
    from finmgr.data import store
    from finmgr.data.benchmarks import read_total_return
    from finmgr.features.liquidity import ADV_COLUMN, read_liquidity

    cutoff = pit.resolve_as_of(as_of)
    resolved = settings or load_settings()
    base = Path(root) if root is not None else resolved.data_dir
    closes = read_total_return(tickers, start=start, as_of=cutoff, settings=resolved, root=root)
    bars = store.read_bars(
        tickers, start=start, as_of=cutoff, settings=resolved, root=base / "bars" / "daily"
    )
    liquidity = read_liquidity(tickers, as_of=cutoff, start=start, settings=resolved, root=root)
    frame = (
        closes[["ticker", "date", "close", "adj_close_eur"]]
        .merge(bars[["ticker", "date", "open"]], on=["ticker", "date"], how="inner")
        .merge(
            liquidity[["ticker", "date", ADV_COLUMN]].rename(
                columns={ADV_COLUMN: LIQUIDITY_COLUMN}
            ),
            on=["ticker", "date"],
            how="left",
        )
    )
    frame[OPEN_COLUMN] = adjusted_open(frame)
    frame = frame.dropna(subset=["adj_close_eur"])
    out = frame[list(BACKTEST_PRICE_COLUMNS)].sort_values(["ticker", "date"], kind="stable")
    return pit.guard(out.reset_index(drop=True), cutoff, what="read_backtest_prices")
