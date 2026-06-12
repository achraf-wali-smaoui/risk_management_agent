
import pandas as pd
import numpy as np
import yfinance as yf
import os
from pathlib import Path


## DATA
def get_ohlcv_from_list(tickers: list[str], timeframe: str, period: str, path: str, save: bool = True) -> pd.DataFrame:
    dfs = []
    for ticker in tickers:
        df = get_ohlcv(ticker, timeframe, period, path, save)
        df["Ticker"] = ticker
        dfs.append(df)
    df = pd.concat(dfs)
    df.set_index("Ticker", inplace=True)
    return df

def get_ohlcv(ticker: str, timeframe: str, period: str, path: str, save: bool = True) -> pd.DataFrame:
    
    if not os.path.exists(f"{path}/{ticker}_{timeframe}_{period}.csv"):
        df = yf.download(
            tickers=ticker,
            interval=timeframe,
            period=period,
            auto_adjust=False,
            progress=False
        )

        # If MultiIndex columns (yfinance case)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure Date is index
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)

        # Ensure columns are single-level and clean
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        if save:
            output_dir = Path(path)
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"{ticker}_{timeframe}_{period}.csv"
            df.to_csv(output_file) 
    else:
        df = pd.read_csv(
            f"{path}/{ticker}_{timeframe}_{period}.csv",
            skiprows=[1, 2],        
            index_col=0,
            parse_dates=[0],        
        )


    df = df[["Open", "High", "Low", "Close", "Volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    df.reset_index(inplace=True)
    df.rename(columns={"Price": "Date"}, inplace=True)

    df.dropna(inplace=True)

    return df


def normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLCV input to a simple DataFrame with columns:
    ["Open","High","Low","Close","Volume"].

    Handles:
    - yfinance MultiIndex columns (e.g., first level "Price", second level "AAPL")
    - lowercase variants
    - extra columns

    This prevents KeyError/validation errors when using real market loaders.
    """
    out = df.copy()

    # If MultiIndex columns: try selecting first ticker, then flatten.
    if isinstance(out.columns, pd.MultiIndex):
        # Common yfinance formats:
        # columns levels like: ("Price","Open"), ("Ticker","AAPL") OR ("Open","AAPL")
        # We'll attempt: if "Open" exists at any level, pick those.
        level_values = [set(map(str, out.columns.get_level_values(i))) for i in range(out.columns.nlevels)]
        # Try to locate OHLCV labels in any level
        ohlcv = ["Open", "High", "Low", "Close", "Volume"]
        found = False
        for lvl in range(out.columns.nlevels):
            if all(x in level_values[lvl] for x in ohlcv):
                # Select columns where that level is OHLCV name.
                cols = {}
                for name in ohlcv:
                    mask = out.columns.get_level_values(lvl).astype(str) == name
                    candidates = out.columns[mask]
                    if len(candidates) == 0:
                        continue
                    # If multiple candidates (multiple tickers), take the first.
                    cols[name] = candidates[0]
                if len(cols) == 5:
                    out = out[list(cols.values())].copy()
                    out.columns = list(cols.keys())
                    found = True
                    break
        if not found:
            # fallback: take first ticker slice by second level if possible
            # and flatten by keeping first level names if they look like OHLCV
            try:
                # take first unique ticker from last level
                tick = out.columns.get_level_values(-1)[0]
                out = out.xs(tick, axis=1, level=-1)
                # now columns might be OHLCV
                out.columns = [str(c) for c in out.columns]
            except Exception as e:
                raise ValueError(f"Could not normalize MultiIndex OHLCV columns: {e}")

    # Normalize column names casing
    rename_map = {}
    for c in out.columns:
        cs = str(c).strip()
        rename_map[c] = cs[0].upper() + cs[1:].lower() if cs else cs
    out = out.rename(columns=rename_map)

    # Ensure required columns exist
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV DataFrame missing columns after normalization: {missing}")

    return out[required].copy()





### CURVE FUNCTIONS
# --------------------------------------------------
# Funciones para calcular Equity Curve y Métricas
# --------------------------------------------------

def equity_curve_from_returns(
    returns: pd.Series,
    *,
    initial_capital: float = 100000.0
) -> pd.Series:
    """
    Calcula la equity curve desde una serie de retornos.
    
    Args:
        returns: Serie de retornos (p.ej. df['Close'].pct_change())
        initial_capital: Capital inicial
    
    Returns:
        Serie con la evolución del capital (equity curve)
    """
    returns = returns.dropna()
    equity = (1 + returns).cumprod() * initial_capital
    return equity


def equity_curve_from_pnl(
    pnl: pd.Series,
    *,
    initial_capital: float = 100000.0
) -> pd.Series:
    """
    Calcula la equity curve desde una serie de PnL (profit and loss).
    
    Args:
        pnl: Serie de PnL por barra
        initial_capital: Capital inicial
    
    Returns:
        Serie con la evolución del capital (equity curve)
    """
    pnl = pnl.dropna()
    equity = initial_capital + pnl.cumsum()
    return equity


def equity_curve_buy_and_hold(
    df_ohlcv: pd.DataFrame,
    *,
    initial_capital: float = 100000.0,
    price_col: str = "Close"
) -> pd.Series:
    """
    Calcula la equity curve de una estrategia buy-and-hold (comprar y mantener).
    
    Args:
        df_ohlcv: DataFrame con datos OHLCV
        initial_capital: Capital inicial
        price_col: Columna de precio a usar (default: "Close")
    
    Returns:
        Serie con la evolución del capital (equity curve)
    """
    prices = df_ohlcv[price_col]
    returns = prices.pct_change().dropna()
    return equity_curve_from_returns(returns, initial_capital=initial_capital)


def risk_manager_metrics(
    equity_curve: pd.Series = None,
    returns: pd.Series = None,
    *,
    rf_rate_annual: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Calcula métricas estándar para evaluar un Risk Manager.
    
    Métricas calculadas:
      - annual_return: Retorno anualizado
      - annual_vol: Volatilidad anualizada
      - sharpe: Ratio de Sharpe
      - sortino: Ratio de Sortino (solo downside risk)
      - max_drawdown: Máximo drawdown
      - calmar: Ratio de Calmar (return / max_drawdown)
      - hit_ratio: Porcentaje de barras ganadoras
    
    Args:
        equity_curve: Serie con la evolución del capital
        returns: Serie de retornos (alternativa a equity_curve)
        rf_rate_annual: Tasa libre de riesgo anual (default: 0.0)
        periods_per_year: Número de períodos por año (252 para diario, ~252*24 para 1h)
    
    Returns:
        Serie con todas las métricas
    """
    if returns is None:
        if equity_curve is None:
            raise ValueError("Debes pasar equity_curve o returns.")
        returns = equity_curve.pct_change().dropna()
    else:
        returns = returns.dropna()

    if returns.empty:
        raise ValueError("Serie de retornos vacía.")

    # Rentabilidad anualizada
    cum_return = (1 + returns).prod() - 1
    n = len(returns)
    ann_return = (1 + cum_return) ** (periods_per_year / n) - 1

    # Volatilidad anualizada
    ann_vol = returns.std(ddof=0) * np.sqrt(periods_per_year)

    # Sharpe
    rf_per_period = rf_rate_annual / periods_per_year
    excess_ret = returns - rf_per_period
    ann_excess = excess_ret.mean() * periods_per_year
    sharpe = ann_excess / ann_vol if ann_vol > 0 else np.nan

    # Sortino (solo downside)
    downside = excess_ret[excess_ret < 0]
    downside_vol = downside.std(ddof=0) * np.sqrt(periods_per_year) if len(downside) > 0 else np.nan
    sortino = ann_excess / downside_vol if downside_vol and downside_vol > 0 else np.nan

    # Máximo drawdown y Calmar
    if equity_curve is None:
        equity_curve = (1 + returns).cumprod()
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_dd = drawdown.min()
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan

    # Hit ratio (porcentaje de barras ganadoras)
    hit_ratio = (returns > 0).mean()

    return pd.Series(
        {
            "annual_return": ann_return,
            "annual_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "hit_ratio": hit_ratio,
        }
    )

