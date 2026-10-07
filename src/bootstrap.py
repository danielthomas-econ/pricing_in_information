import numpy as np
import pandas as pd

from common import binary_entropy

# bootstrap standard errors for the mutual information estimates (q1 and q2)

# we resample markets, not market hours, as they're the independent units.

# resampling 43m rows 1000 times would take forever, so we collapse the panel
# down to one row per market first. the estimator only needs each market's
# outcome and how its weight is spread over the 20 price bins, so after that
# each bootstrap draw is just "how many times was each market picked", done
# using a multinominal draw.


# --------------------------------------------------------- - -------------------------------------------------------- #
#### QUESTION 1
# --------------------------------------------------------- - -------------------------------------------------------- #

# squashes the panel down to one row per market
# weighting and drop_last_hours work exactly like in entropy_calculations
def collapse_markets(input, weighting="market", drop_last_hours=0):
    if drop_last_hours > 0:
        df = input[input["time_to_resolution"] > drop_last_hours]
    else:
        df = input

    # integer codes for markets (0 to N-1) and price bins (0 to 19)
    # cat.codes gives -1 for rows with no price bin
    market_codes, market_ids = pd.factorize(df.index.get_level_values("market_id"))
    bin_codes = df["price_bins"].cat.codes.to_numpy().astype(np.int64)
    N = len(market_ids)
    K = len(df["price_bins"].cat.categories)

    # hours per market, same as transform("size") in entropy_calculations
    hours = np.bincount(market_codes, minlength=N).astype(float)

    # hours each market spent in each price bin, an N x K matrix
    # market i in bin b gets flattened to position i*K + b, counted, then reshaped back
    has_bin = bin_codes >= 0
    hours_in_bin = np.bincount(market_codes[has_bin] * K + bin_codes[has_bin],
                               minlength=N * K).reshape(N, K).astype(float)

    # outcome of each market (its the same on every row so any row works)
    outcome = np.zeros(N)
    outcome[market_codes] = df["yes_outcomes"].to_numpy().astype(float)

    # market weighting: each hour gets 1/hours, so the bin weights of a market sum to 1
    # hour weighting: each hour counts once, so a market's total weight is its hours
    if weighting == "market":
        bin_weights = hours_in_bin / hours[:, None]
        market_weight = np.ones(N)
    elif weighting == "hour":
        bin_weights = hours_in_bin
        market_weight = hours

    return market_ids, outcome, bin_weights, market_weight


# the estimator from entropy_calculations, just rewritten so it can run on many draws at once
# counts is (draws x markets): how many times each market got picked in each draw
# counts of all ones gives back the original estimate
def mi_from_counts(counts, outcome, bin_weights, market_weight):
    counts = np.atleast_2d(counts).astype(float)

    #### CALCULATING ENTROPY H(Y)
    W = counts @ market_weight
    p_yes = (counts @ (market_weight * outcome)) / W
    entropy_Y = binary_entropy(p_yes)

    #### CALCULATING CONDITIONAL ENTROPY H(Y|P)
    # total weights and yes-weights in each bin, for every draw
    w_bin = counts @ bin_weights
    wy_bin = counts @ (bin_weights * outcome[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        yes_share = np.where(w_bin > 0, wy_bin / w_bin, 0.0)
    cond_entropy = (w_bin / W[:, None] * binary_entropy(yes_share)).sum(axis=1)

    #### MILLER MADOW CORRECTION
    # N = markets in the draw, K = bins that have both yes and no outcomes
    N = counts.sum(axis=1)
    K = ((yes_share > 0) & (yes_share < 1)).sum(axis=1)
    bias = (K - 1) / (2 * N * np.log(2))
    MI_mm = entropy_Y - cond_entropy - bias

    uncertainty_reduced = MI_mm * 100 / entropy_Y
    return entropy_Y, cond_entropy, MI_mm, uncertainty_reduced


# picks N markets with replacement, B times
# if clusters is given (eg event id), we pick whole clusters instead and every
# market in a cluster gets the cluster's count
def draw_counts(rng, N, B, clusters=None):
    if clusters is None:
        return rng.multinomial(N, np.full(N, 1 / N), size=B).astype(np.float32)

    cluster_codes, cluster_ids = pd.factorize(np.asarray(clusters))
    G = len(cluster_ids)
    cluster_counts = rng.multinomial(G, np.full(G, 1 / G), size=B).astype(np.float32)
    return cluster_counts[:, cluster_codes]


# the actual bootstrap for q1
# subset: optional True/False array over markets (eg negRisk). if given, we resample
# the full sample but only count markets in the subset, so negrisk and non negrisk
# use the same draws and we can get an SE for their difference too
def bootstrap_entropy(collapsed, B=1000, seed=0, clusters=None, subset=None, chunk=50):
    market_ids, outcome, bin_weights, market_weight = collapsed
    N = len(outcome)
    rng = np.random.default_rng(seed)

    if subset is None:
        keep = np.ones(N)
    else:
        keep = np.asarray(subset).astype(float)

    estimate = mi_from_counts(keep, outcome, bin_weights, market_weight)

    # running in chunks so the counts matrix doesnt eat all the ram
    # (1000 x 50k floats would be 200mb)
    draws = []
    for start in range(0, B, chunk):
        counts = draw_counts(rng, N, min(chunk, B - start), clusters) * keep
        draws.append(np.column_stack(mi_from_counts(counts, outcome, bin_weights, market_weight)))
    draws = np.vstack(draws)

    # turning both into dataframes so theyre easier to read
    names = ["entropy_Y", "cond_entropy", "MI_mm", "uncertainty_reduced"]
    estimate = pd.Series([e[0] for e in estimate], index=names)
    draws = pd.DataFrame(draws, columns=names)

    se = draws.std()
    ci_low, ci_high = draws["uncertainty_reduced"].quantile([0.025, 0.975])
    print(f"Bootstrap: {int(keep.sum()):,} markets, {B:,} draws")
    for label, col in [("Entropy H(Y)", "entropy_Y"),
                       ("Conditional entropy H(Y|P)", "cond_entropy"),
                       ("Corrected I(Y;P)", "MI_mm")]:
        print(f"  {label:<32}{estimate[col]:>9.5f} bits  (SE {se[col]:.5f})")
    print(f"  {'Uncertainty resolved by prices':<32}{estimate['uncertainty_reduced']:>8.2f} %"
          f"     (SE {se['uncertainty_reduced']:.2f})")
    print(f"  {'95% confidence interval':<32}[{ci_low:.2f}%, {ci_high:.2f}%]")
    print()

    return estimate, draws


# --------------------------------------------------------- - -------------------------------------------------------- #
#### QUESTION 2
# --------------------------------------------------------- - -------------------------------------------------------- #

# same cohort and snapshot rules as information_curve, but instead of computing MI
# straight away we keep a markets x days matrix of price bins
# bins[i, d-1] = price bin of market i, d days before close (-1 if no price that day)
def cohort_snapshots(data, min_days, n_bins=20, buffer_hours=48):
    lifetime = data.groupby("market_id", observed=True)["time_to_resolution"].transform("max")
    df = data.loc[lifetime >= 24 * min_days + buffer_hours,
                  ["p", "yes_outcomes", "time_to_resolution"]].copy()

    df["days_before"] = (df["time_to_resolution"] // 24).astype("int32")
    df = df[(df["days_before"] >= 1) & (df["days_before"] <= min_days)]
    snaps = df.groupby(["market_id", "days_before"], observed=True).tail(1)
    del df

    edges = np.linspace(0, 1, n_bins + 1)
    price_bin = pd.cut(snaps["p"], bins=edges, labels=False, include_lowest=True).fillna(-1).astype(int)

    market_codes, market_ids = pd.factorize(snaps.index.get_level_values("market_id"))
    bins = np.full((len(market_ids), min_days), -1, dtype=np.int64)
    bins[market_codes, snaps["days_before"].to_numpy() - 1] = price_bin.to_numpy()

    outcome = np.zeros(len(market_ids))
    outcome[market_codes] = snaps["yes_outcomes"].to_numpy().astype(float)
    return market_ids, outcome, bins


# MI and % resolved at every horizon for every draw, same formulas as information_curve
# each market counts once at each horizon so no weights needed here
def curve_from_counts(counts, outcome, bins, n_bins=20):
    counts = np.atleast_2d(counts).astype(float)
    n_draws, n_days = counts.shape[0], bins.shape[1]
    MI = np.empty((n_draws, n_days))
    pct = np.empty((n_draws, n_days))

    for d in range(n_days):
        # one hot matrix of which bin each market is in on this day
        has_price = bins[:, d] >= 0
        in_bin = np.zeros((len(outcome), n_bins))
        in_bin[np.flatnonzero(has_price), bins[has_price, d]] = 1.0

        N = counts @ has_price.astype(float)
        p_yes = (counts @ (has_price * outcome)) / N
        entropy_Y = binary_entropy(p_yes)

        n_bin = counts @ in_bin
        yes_bin = counts @ (in_bin * outcome[:, None])
        with np.errstate(divide="ignore", invalid="ignore"):
            yes_share = np.where(n_bin > 0, yes_bin / n_bin, 0.0)
        cond_entropy = (n_bin / N[:, None] * binary_entropy(yes_share)).sum(axis=1)

        K = ((yes_share > 0) & (yes_share < 1)).sum(axis=1)
        MI[:, d] = entropy_Y - cond_entropy - (K - 1) / (2 * N * np.log(2))
        pct[:, d] = 100 * MI[:, d] / entropy_Y

    return MI, pct


# bootstrap for one cohort's curve
# each draw resamples whole markets, so the same resampled markets show up at
# every horizon (just like the actual estimate). this means we can also get SEs
# for anything built from the curve, like the gain per day between horizons
def bootstrap_curve(data, min_days, B=1000, seed=0, clusters=None, chunk=100):
    market_ids, outcome, bins = cohort_snapshots(data, min_days)
    N = len(outcome)
    rng = np.random.default_rng(seed)

    # clusters is a series of market_id -> cluster id, lined up with this cohort's markets
    if clusters is not None:
        clusters = pd.Series(clusters).reindex(market_ids).to_numpy()

    MI, pct = curve_from_counts(np.ones(N), outcome, bins)

    MI_draws, pct_draws = [], []
    for start in range(0, B, chunk):
        counts = draw_counts(rng, N, min(chunk, B - start), clusters)
        m, p = curve_from_counts(counts, outcome, bins)
        MI_draws.append(m)
        pct_draws.append(p)
    MI_draws = np.vstack(MI_draws)
    pct_draws = np.vstack(pct_draws)

    curve = pd.DataFrame({
        "n_markets": (bins >= 0).sum(axis=0),
        "mutual_info": MI[0],
        "mutual_info_se": MI_draws.std(axis=0, ddof=1),
        "pct_resolved": pct[0],
        "pct_resolved_se": pct_draws.std(axis=0, ddof=1),
        "ci_low": np.quantile(pct_draws, 0.025, axis=0),
        "ci_high": np.quantile(pct_draws, 0.975, axis=0),
    }, index=pd.Index(np.arange(1, min_days + 1), name="days_before"))

    return curve, pct_draws


# gain per day over each stretch (same stretches as plot_daily_gains), with SEs
# pct_draws column d-1 is d days before close
def gains_per_day(curve, pct_draws, stretches=[(120, 60), (60, 30), (30, 14), (14, 7), (7, 4), (4, 2), (2, 1)]):
    pct = curve["pct_resolved"]
    rows = []
    for far, near in stretches:
        # skip stretches longer than the cohort
        if far > len(pct):
            continue
        gain = (pct.loc[near] - pct.loc[far]) / (far - near)
        gain_draws = (pct_draws[:, near - 1] - pct_draws[:, far - 1]) / (far - near)

        rows.append({"stretch": f"{near}–{far} days", "gain_per_day": gain,
                     "se": gain_draws.std(ddof=1),
                     "ci_low": np.quantile(gain_draws, 0.025),
                     "ci_high": np.quantile(gain_draws, 0.975)})

    return pd.DataFrame(rows).set_index("stretch")
