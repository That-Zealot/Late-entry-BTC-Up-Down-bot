#!/usr/bin/env python3
"""
Polymarket 5-minute BTC "Up or Down" late-entry bot.

Rules (all configurable):
  ENTRY : only in the final 30 seconds of a window, buy the side whose best ASK
          is between 0.98 and 0.99, spending up to $20
  EXIT  : none. No stop loss, no loss limit. Hold to settlement ($1.00/share if right, $0 if wrong)
  Max one entry per 5-minute window.

Modes:
  default (dry run) - reads REAL live Polymarket order books, simulates the fill against
                      the actual book depth, tracks the real resolution, places NO orders,
                      needs NO keys. This is a forward test: run it for days, then look at trades.csv.
  --live            - places real orders through py-clob-client.

Usage:
  pip install requests               # dry run
  pip install requests py-clob-client   # live
  python updown_bot.py                                  # dry run, forever (Ctrl-C prints a summary)
  POLY_PRIVATE_KEY=0x... python updown_bot.py --live --yes-i-accept-the-risk
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("updown")

WINDOW = 300  # seconds
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
MAX_PENDING_WINDOWS = 24  # keep checking an unresolved position for ~2 hours, then flag it


# ------------------------------------------------------------------ models ---
@dataclass
class Config:
    entry_min: float = 0.98          # buy when best ask >= this ...
    entry_max: float = 0.99          # ... and <= this (never pay more)
    stake_usd: float = 20.0          # USD spent per entry (at most)
    poll_s: float = 0.5              # book polling interval inside the entry window
    enter_last_s: float = 30.0       # only enter during the final N seconds of a window
    enter_before_end_s: float = 3.0  # ...but not in the last N seconds (too late to fill)
    max_slippage: float = 0.01       # buy limit = ask + slippage, capped at entry_max
    assumed_fee_per_share: float = 0.0  # dry run only: check Polymarket's real fee schedule
    trades_csv: str = "trades.csv"


@dataclass
class Book:
    bids: list  # [(price, size)]
    asks: list

    @property
    def best_bid(self) -> Optional[float]:
        return max((p for p, _ in self.bids), default=None)

    @property
    def best_ask(self) -> Optional[float]:
        return min((p for p, _ in self.asks), default=None)


@dataclass
class Market:
    slug: str
    start_ts: int
    end_ts: int
    tokens: dict  # {"up": token_id, "down": token_id}


@dataclass
class Fill:
    shares: float
    avg_price: float


@dataclass
class Position:
    market: Market
    side: str
    shares: float
    cost: float
    ask_seen: float
    age: int = 0  # windows spent waiting for resolution


class RealClock:
    def now(self): return time.time()
    def sleep(self, s): time.sleep(max(s, 0.0))


# -------------------------------------------------------------- Polymarket ---
class Polymarket:
    """Everything Polymarket-specific: market discovery (Gamma), order books and
    resolution (public endpoints, no key), and order placement (py-clob-client, live only).

    Not tested against the live API by the author. Start with a small stake.
    Check that you're eligible to use Polymarket where you live.
    Live env: POLY_PRIVATE_KEY, optional POLY_FUNDER (proxy/Safe address), POLY_SIG_TYPE (0/1/2).
    """

    def __init__(self, live=False, session=None, assumed_fee=0.0):
        self.live, self.fee = live, assumed_fee
        if session is None:
            import requests
            session = requests.Session()
        self.http = session
        self.client = None
        if live:
            from py_clob_client.client import ClobClient
            self.client = ClobClient(
                CLOB,
                key=os.environ["POLY_PRIVATE_KEY"],
                chain_id=137,
                signature_type=int(os.environ.get("POLY_SIG_TYPE", "0")),
                funder=os.environ.get("POLY_FUNDER"),
            )
            self.client.set_api_creds(self.client.create_or_derive_api_creds())

    @staticmethod
    def _jl(x):
        return json.loads(x) if isinstance(x, str) else x

    def _gamma_market(self, slug):
        r = self.http.get(f"{GAMMA}/markets", params={"slug": slug}, timeout=5)
        r.raise_for_status()
        data = r.json()
        return data[0] if data else None

    def market_for(self, start_ts: int) -> Optional[Market]:
        slug = f"btc-updown-5m-{start_ts}"
        mk = self._gamma_market(slug)
        if not mk:
            return None
        by_name = {o.strip().lower(): t for o, t in zip(self._jl(mk["outcomes"]), self._jl(mk["clobTokenIds"]))}
        return Market(slug, start_ts, start_ts + WINDOW, {"up": by_name["up"], "down": by_name["down"]})

    @staticmethod
    def _parse_book(b) -> Book:
        return Book([(float(x["price"]), float(x["size"])) for x in b.get("bids", [])],
                    [(float(x["price"]), float(x["size"])) for x in b.get("asks", [])])

    def books(self, m: Market) -> dict:
        """Both sides' order books in one request (falls back to one request per side)."""
        ids = list(m.tokens.values())
        try:
            r = self.http.post(f"{CLOB}/books", json=[{"token_id": t} for t in ids], timeout=5)
            r.raise_for_status()
            raw = {b["asset_id"]: b for b in r.json()}
            if set(ids) - set(raw):
                raise KeyError("incomplete batch")
        except Exception:
            raw = {}
            for t in ids:
                r = self.http.get(f"{CLOB}/book", params={"token_id": t}, timeout=5)
                r.raise_for_status()
                raw[t] = r.json()
        return {side: self._parse_book(raw[tok]) for side, tok in m.tokens.items()}

    def buy(self, m: Market, side: str, usd: float, limit: float, book: Book) -> Optional[Fill]:
        if not self.live:
            return self._simulate_buy(book, usd, limit)
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY
        # Limit prices with 2 decimals are valid on any tick size, including finer ticks near the extremes.
        size = math.floor(usd / limit * 100) / 100
        args = OrderArgs(token_id=m.tokens[side], price=round(limit, 2), size=size, side=BUY)
        resp = self.client.post_order(self.client.create_order(args), OrderType.FAK)  # fill-and-kill
        if not resp.get("success"):
            log.warning("buy rejected: %s", resp)
            return None
        status = str(resp.get("status", "")).lower()
        try:  # BUY: makingAmount = USDC spent, takingAmount = shares received
            spent, shares = float(resp.get("makingAmount") or 0), float(resp.get("takingAmount") or 0)
        except ValueError:
            raise RuntimeError(f"Unrecognised order response, stopping. Check your positions! {resp!r}")
        if shares <= 0:
            if status == "matched":
                raise RuntimeError(f"Order matched but amounts missing, stopping. Check your positions! {resp!r}")
            return None
        return Fill(shares, spent / shares)

    def _simulate_buy(self, book: Book, usd: float, limit: float) -> Optional[Fill]:
        """Dry run: walk the real ask ladder up to the limit price, spending at most `usd`."""
        spent = shares = 0.0
        for price, size in sorted(book.asks):
            if price > limit + 1e-9 or spent >= usd - 1e-9:
                break
            take = min(size, (usd - spent) / price)
            shares += take
            spent += take * price
        if shares <= 0:
            return None
        return Fill(shares, (spent + shares * self.fee) / shares)

    def settle(self, m: Market, side: str) -> Optional[float]:
        """1.0 / 0.0 once the market is closed and one outcome price is ~1.0, else None."""
        mk = self._gamma_market(m.slug)
        if not mk or not mk.get("closed"):
            return None
        prices = {o.strip().lower(): float(p)
                  for o, p in zip(self._jl(mk["outcomes"]), self._jl(mk["outcomePrices"]))}
        if max(prices.values()) < 0.99 or side not in prices:
            return None
        return 1.0 if prices[side] >= 0.99 else 0.0


# --------------------------------------------------------------------- bot ---
class Bot:
    def __init__(self, pm: Polymarket, cfg: Config, clock=None):
        self.pm, self.cfg = pm, cfg
        self.clock = clock or RealClock()
        self.pending: list = []
        self.realized = 0.0
        self.stats = dict(windows=0, entries=0, won=0, lost=0)

    def run(self, max_windows: Optional[int] = None) -> None:
        log.info("%s mode | stake $%.2f | entry %.2f-%.2f in the last %.0fs | no stop, no loss limit",
                 "LIVE" if self.pm.live else "DRY-RUN", self.cfg.stake_usd,
                 self.cfg.entry_min, self.cfg.entry_max, self.cfg.enter_last_s)
        n = 0
        while max_windows is None or n < max_windows:
            self._run_window(int(self.clock.now() // WINDOW * WINDOW))
            n += 1
            self.stats["windows"] = n

    def _run_window(self, start: int) -> None:
        cfg, clk = self.cfg, self.clock
        end = start + WINDOW
        market: Optional[Market] = None
        pos: Optional[Position] = None

        while clk.now() < end:
            try:
                if market is None:
                    market = self.pm.market_for(start)
                    if market is None:
                        clk.sleep(1.0)
                        continue
                remaining = end - clk.now()
                if pos is not None:
                    clk.sleep(remaining)                          # holding: wait for the window to end
                    continue
                if remaining > cfg.enter_last_s:
                    clk.sleep(remaining - cfg.enter_last_s)       # entry window not open yet
                    continue
                if remaining < cfg.enter_before_end_s:
                    clk.sleep(remaining)                          # too late to enter
                    continue
                pos = self._try_enter(market)
                if pos is not None:
                    continue
            except RuntimeError:
                raise  # unrecoverable (e.g. unrecognised order response): stop the bot
            except Exception as e:  # transient API error: log and keep going
                log.warning("window %s: %s: %s", start, type(e).__name__, e)
            clk.sleep(cfg.poll_s)

        if pos is not None:
            self.pending.append(pos)
        self._resolve_pending()

    def _try_enter(self, m: Market) -> Optional[Position]:
        cfg = self.cfg
        books = self.pm.books(m)
        best = None
        for side, bk in books.items():
            ask = bk.best_ask
            if ask is not None and cfg.entry_min - 1e-9 <= ask <= cfg.entry_max + 1e-9:
                if best is None or ask > best[1]:
                    best = (side, ask)
        if best is None:
            return None
        side, ask = best
        limit = round(min(ask + cfg.max_slippage, cfg.entry_max), 2)
        fill = self.pm.buy(m, side, cfg.stake_usd, limit, books[side])
        if fill is None or fill.shares <= 0:
            log.info("buy %s not filled (ask %.3f, limit %.2f)", side, ask, limit)
            return None
        self.stats["entries"] += 1
        log.info("BUY  %-4s %.2f sh @ %.4f = $%.2f (ask was %.3f, %.0fs left)", side, fill.shares,
                 fill.avg_price, fill.shares * fill.avg_price, ask, m.end_ts - self.clock.now())
        return Position(m, side, fill.shares, fill.shares * fill.avg_price, ask)

    def _resolve_pending(self) -> None:
        keep = []
        for pos in self.pending:
            try:
                payout = self.pm.settle(pos.market, pos.side)
            except Exception as e:
                log.warning("settle %s: %s: %s", pos.market.slug, type(e).__name__, e)
                payout = None
            if payout is None:
                pos.age += 1
                if pos.age > MAX_PENDING_WINDOWS:
                    log.error("%s still unresolved after %d windows. Check it manually.", pos.market.slug, pos.age)
                else:
                    keep.append(pos)
                continue
            pnl = payout * pos.shares - pos.cost
            self.stats["won" if payout > 0 else "lost"] += 1
            self.realized += pnl
            log.info("window %s %-4s settled %-4s P&L %+.2f | total %+.2f", pos.market.start_ts, pos.side,
                     "WIN" if payout > 0 else "LOSS", pnl, self.realized)
            self._record(pos, payout, pnl)
        self.pending = keep

    def _record(self, pos: Position, payout: float, pnl: float) -> None:
        path = self.cfg.trades_csv
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["logged_utc", "window_start", "side", "ask_seen", "avg_fill", "shares",
                            "cost_usd", "payout_per_share", "pnl_usd", "mode"])
            w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), pos.market.start_ts, pos.side,
                        f"{pos.ask_seen:.4f}", f"{pos.cost / pos.shares:.4f}", f"{pos.shares:.4f}",
                        f"{pos.cost:.2f}", payout, f"{pnl:.2f}", "live" if self.pm.live else "dry"])

    def summary(self) -> str:
        s = self.stats
        done = s["won"] + s["lost"]
        wr = f"{s['won'] / done:.1%}" if done else "n/a"
        avg = self.realized / done if done else 0.0
        return (f"windows={s['windows']} entries={s['entries']} won={s['won']} lost={s['lost']} "
                f"win rate={wr} pending={len(self.pending)} | net P&L ${self.realized:+.2f} (${avg:+.3f}/trade)")


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="place real orders (default is a dry run on live data)")
    ap.add_argument("--yes-i-accept-the-risk", action="store_true")
    ap.add_argument("--windows", type=int, default=None, help="stop after N windows (default: run until Ctrl-C)")
    ap.add_argument("--stake", type=float, default=20.0, help="USD per entry (default 20)")
    ap.add_argument("--entry-min", type=float, default=0.98)
    ap.add_argument("--entry-max", type=float, default=0.99)
    ap.add_argument("--entry-window", type=float, default=30.0, help="only enter in the last N seconds")
    ap.add_argument("--assumed-fee", type=float, default=0.0, help="dry run only: fee per share to assume")
    ap.add_argument("--csv", default="trades.csv", help="where resolved trades are logged")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if a.live and not a.yes_i_accept_the_risk:
        sys.exit("Refusing to trade live without --yes-i-accept-the-risk")

    cfg = Config(entry_min=a.entry_min, entry_max=a.entry_max, enter_last_s=a.entry_window,
                 stake_usd=a.stake, assumed_fee_per_share=a.assumed_fee, trades_csv=a.csv)
    bot = Bot(Polymarket(live=a.live, assumed_fee=a.assumed_fee), cfg)
    try:
        bot.run(a.windows)
    except KeyboardInterrupt:
        print("\ninterrupted")
    print(bot.summary())


if __name__ == "__main__":
    main()
