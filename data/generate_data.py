"""Generate a SIMULATED cohort of MDR/RR-TB patients on treatment in western Kenya.

!! SIMULATED DATA !!
Every record in this file is produced by a random-number generator with assumed,
illustrative effect sizes. It does NOT reproduce, approximate or derive from the data or
findings of any real study (including Okumu A, et al., J Clin Tuberc Other Mycobact Dis
2024;37:100466). No real patient data are used anywhere in this repository.

Design (so that there is realistic structure for the analysis to find)
---------------------------------------------------------------------
* ~1,200 adults with bacteriologically confirmed, baseline culture-positive MDR/RR-TB start
  treatment at clinics in six western Kenya counties between Jan 2023 and Dec 2025.
* Follow-up runs from treatment start to 24 months, death, loss to follow-up (LTFU),
  transfer out, or the data cut-off on 30 Jun 2026 (administrative censoring), whichever
  comes first. Late enrolees therefore have shorter follow-up, exactly as in a real cohort.
* Three latent event times are drawn for each patient from CAUSE-SPECIFIC hazards:
    - culture conversion  Weibull (shape 1.3), slowed by high smear grade, cavitation,
                          pre-XDR resistance, diabetes, undernutrition, smoking, previous
                          treatment and low adherence; the short all-oral regimen speeds
                          conversion strongly in the first 120 days only (a deliberate
                          NON-proportional effect for the PH-checking section to find).
    - death               Weibull (shape 0.8, i.e. highest risk early), raised by age,
                          advanced HIV (CD4 < 200), no ART, undernutrition, diabetes,
                          pre-XDR, heavy bacillary load; the hazard drops by 60% after a
                          patient converts.
    - LTFU                constant hazard while on treatment only (6 months short regimen,
                          18 months longer regimen), raised by distance to clinic, low DOT
                          adherence, smoking, male sex and young age.
  Conversion can only be detected at a monthly sputum-culture visit (day ~30, 60, 90 ... +/- 3
  days), so recorded conversion dates cluster on visit days (heavily tied times), as in
  programme data. Observed outcomes are the FIRST of these times, plus random transfer-out
  and administrative censoring. Death is still tracked after conversion (overall survival).
* Missing data: CD4 is missing for ~25% of HIV-positive patients (more often for patients
  living far from the clinic, i.e. missing at random given distance); BMI is missing for
  ~4%. A few data-quality problems are planted for the cleaning step: duplicated records
  and impossible BMI values.

Output: data/mdr_tb_cohort_simulated.csv (one row per patient record, dates not times,
like a programme register export).

Usage: python data/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20240466)
OUT = Path(__file__).parent
N = 1200
START, END_ENROL, CUTOFF = pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-31"), pd.Timestamp("2026-06-30")
MAX_FU = 730  # days (24 months)

COUNTIES = {  # share, median distance to clinic (km)
    "Kisumu": (.24, 6), "Siaya": (.17, 12), "Homa Bay": (.18, 15),
    "Migori": (.15, 16), "Kakamega": (.16, 11), "Busia": (.10, 13)}


def piecewise_weibull(e, scale, shape, mult_before, mult_after, cut):
    """Invert a Weibull cumulative hazard whose multiplier changes at time `cut`.

    H(t) = m1 * (t/scale)^k                                   for t <= cut
         = m1 * (cut/scale)^k + m2 * ((t/scale)^k - (cut/scale)^k)   for t > cut
    `e` is an Exp(1) draw; returns t such that H(t) = e (vectorised).
    """
    h_cut = (cut / scale) ** shape
    early = e <= mult_before * h_cut
    t = np.empty_like(e)
    t[early] = scale * (e[early] / mult_before[early]) ** (1 / shape)
    late = ~early
    h_t = h_cut[late] + (e[late] - mult_before[late] * h_cut[late]) / mult_after[late]
    t[late] = scale * h_t ** (1 / shape)
    return t


# ------------------------------------------------------------------ baseline covariates
cn = list(COUNTIES)
county = RNG.choice(cn, N, p=[COUNTIES[c][0] for c in cn])
enrol = START + pd.to_timedelta(RNG.integers(0, (END_ENROL - START).days + 1, N), unit="D")
age = np.clip(np.round(RNG.gamma(9, 4.3, N)), 18, 80).astype(int)
male = RNG.random(N) < .64
hiv = RNG.random(N) < np.where(male, .27, .42)
cd4 = np.where(hiv, np.clip(np.round(RNG.lognormal(np.log(260), .75, N)), 5, 1400), np.nan)
on_art = np.where(hiv, (RNG.random(N) < .88).astype(float), np.nan)
distance = np.round(np.clip(np.array([COUNTIES[c][1] for c in county]) * RNG.lognormal(0, .6, N), .5, 120), 1)
bmi = np.round(np.clip(RNG.normal(19.6 - .9 * hiv + .4 * male, 2.6, N), 12.5, 34), 1)
diabetes = RNG.random(N) < 1 / (1 + np.exp(-(-3.6 + .045 * (age - 38))))
smoker = RNG.random(N) < np.where(male, .34, .05)
prev_tb = RNG.random(N) < .52
cavitary = RNG.random(N) < np.where(hiv & (np.nan_to_num(cd4, nan=999) < 200), .25, .50)
smear_p = np.where(cavitary[:, None], [[.18, .30, .52]], [[.42, .32, .26]])
smear = np.array([RNG.choice(["Negative/scanty", "1+", "2+/3+"], p=p) for p in smear_p])
pre_xdr = RNG.random(N) < np.where(prev_tb, .19, .09)
year = enrol.year.to_numpy()
p_short = 1 / (1 + np.exp(-(-.3 + 1.1 * (year - 2023) - .9 * pre_xdr - .6 * prev_tb)))
short = RNG.random(N) < p_short
adherence = np.round(np.clip(100 * RNG.beta(14 - 2.5 * np.log1p(distance / 10) - 3 * smoker, 1.6, N), 40, 100))

# ------------------------------------------------------------------ latent event times
undernut = bmi < 18.5
smear2 = smear == "2+/3+"
smear1 = smear == "1+"
low_cd4 = hiv & (np.nan_to_num(cd4, nan=999) < 200)
adh_def10 = (90 - adherence) / 10       # per 10 percentage points BELOW 90%

# (1) Culture conversion: log cause-specific hazard ratios
lp_c = (np.log(.72) * smear1 + np.log(.52) * smear2 + np.log(.70) * cavitary + np.log(.62) * pre_xdr
        + np.log(.78) * diabetes + np.log(.80) * undernut + np.log(.85) * smoker + np.log(.85) * prev_tb
        + np.log(.85) * adh_def10 + np.log(.88) * low_cd4)
m1 = np.exp(lp_c) * np.where(short, 2.0, 1.0)     # short regimen: HR 2.0 in first 120 days ...
m2 = np.exp(lp_c) * np.where(short, 1.0, 1.0)     # ... and no difference afterwards
t_conv = piecewise_weibull(RNG.exponential(size=N), 85.0, 1.3, m1, m2, np.full(N, 120.0))

# (2) Death: hazard falls 60% once the patient has converted
lp_d = (np.log(1.30) * (age - 38) / 10 + np.log(1.20) * male + np.log(1.4) * (hiv & ~low_cd4)
        + np.log(3.0) * low_cd4 + np.log(1.7) * (on_art == 0) + np.log(2.0) * undernut
        + np.log(1.5) * diabetes + np.log(1.7) * pre_xdr + np.log(1.25) * cavitary + np.log(1.4) * smear2)
mult_d = np.exp(lp_d)
t_death = piecewise_weibull(RNG.exponential(size=N), 1.2e4, .8, mult_d, .4 * mult_d, t_conv)

# (3) LTFU: constant hazard while on treatment only
tx_days = np.where(short, 183, 548)
lp_l = (np.log(1.18) * distance / 10 + np.log(1.55) * adh_def10 + np.log(1.4) * smoker
        + np.log(1.3) * male + np.log(1.5) * (age < 25))
t_ltfu = RNG.exponential(1 / (np.exp(lp_l) / 3500), N)
t_ltfu[t_ltfu > tx_days] = np.inf

t_transfer = np.where(RNG.random(N) < .03, RNG.uniform(10, MAX_FU, N), np.inf)
t_admin = np.minimum(MAX_FU, (CUTOFF - enrol).days.to_numpy()).astype(float)

# round latent times up to whole days (registers record dates)
t_death, t_ltfu, t_transfer = (np.ceil(x) for x in (t_death, t_ltfu, t_transfer))
# Conversion is only DETECTED at monthly sputum-culture visits (day ~30, 60, 90, ... +/- 3 days):
# the recorded date is the first scheduled culture after the (unobservable) biological conversion.
t_conv = 30 * np.ceil(t_conv / 30) + RNG.integers(-3, 4, N)
end_obs = np.minimum.reduce([t_death, t_ltfu, t_transfer, t_admin])     # last day vital status is known
conv_seen = t_conv < end_obs                                            # strict: converted before leaving
death_seen = t_death <= np.minimum.reduce([t_ltfu, t_transfer, t_admin])
ltfu_seen = (t_ltfu == end_obs) & ~death_seen
transfer_seen = (t_transfer == end_obs) & ~death_seen & ~ltfu_seen


def as_date(days, mask):
    d = pd.Series(enrol + pd.to_timedelta(np.where(mask, days, 0), unit="D")).dt.strftime("%Y-%m-%d")
    return d.where(mask, "")


df = pd.DataFrame({
    "patient_id": [f"DRTB-{i:04d}" for i in range(1, N + 1)],
    "county": county,
    "treatment_start_date": pd.Series(enrol).dt.strftime("%Y-%m-%d"),
    "age_years": age,
    "sex": np.where(male, "Male", "Female"),
    "hiv_status": np.where(hiv, "Positive", "Negative"),
    "cd4_count": cd4,
    "on_art": pd.Series(on_art).map({1.0: "Yes", 0.0: "No"}).fillna(""),
    "bmi_kg_m2": bmi,
    "diabetes": np.where(diabetes, "Yes", "No"),
    "current_smoker": np.where(smoker, "Yes", "No"),
    "previous_tb_treatment": np.where(prev_tb, "Yes", "No"),
    "cavitary_disease": np.where(cavitary, "Yes", "No"),
    "baseline_smear_grade": smear,
    "resistance_pattern": np.where(pre_xdr, "Pre-XDR", "RR/MDR"),
    "regimen": np.where(short, "Short all-oral (6 mo)", "Longer individualised (18-20 mo)"),
    "dot_adherence_pct": adherence.astype(int),
    "distance_to_clinic_km": distance,
    "culture_conversion_date": as_date(t_conv, conv_seen),
    "death_date": as_date(t_death, death_seen),
    "ltfu_date": as_date(t_ltfu, ltfu_seen),
    "transfer_out_date": as_date(t_transfer, transfer_seen),
    "last_followup_date": as_date(end_obs, np.ones(N, bool)),
})

# ------------------------------------------------------------------ missing data & data-quality issues
cd4_miss = hiv & (RNG.random(N) < np.clip(.12 + .012 * distance, 0, .6))   # MAR given distance
df.loc[cd4_miss, "cd4_count"] = np.nan
df.loc[RNG.random(N) < .04, "bmi_kg_m2"] = np.nan
bad = RNG.choice(np.flatnonzero(df.bmi_kg_m2.notna()), 3, replace=False)
df.loc[bad, "bmi_kg_m2"] = [0.0, 98.0, 214.0]                                # impossible entries
dups = df.sample(4, random_state=7)                                         # double-entered records
df = pd.concat([df, dups]).sort_values(["treatment_start_date", "patient_id"]).reset_index(drop=True)

df.to_csv(OUT / "mdr_tb_cohort_simulated.csv", index=False)
print(f"Wrote {len(df):,} rows ({N:,} unique SIMULATED patients) to data/mdr_tb_cohort_simulated.csv")
print(f"converted {conv_seen.sum()} | died before conversion {(death_seen & ~conv_seen).sum()} | "
      f"LTFU before conversion {(ltfu_seen & ~conv_seen).sum()} | deaths overall {death_seen.sum()} | "
      f"short regimen {short.mean():.0%} | HIV+ {hiv.mean():.0%} | BMI<18.5 {undernut.mean():.0%}")
