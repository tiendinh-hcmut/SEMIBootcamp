"""
Section 4.2 — Operator Monitoring Dashboard (Streamlit)
Group 3 · SEMI TECH Zoomers Challenge @ SEMIEXPO Viet Nam 2026
Maintainer: Hua Tue Minh

Design: high-performance HMI (ISA-101 spirit). The screen stays neutral grey while
the bay is normal; colour appears only for an abnormal state, and always next to a
word that names it. One accent series (the proposed EquipState_MPC), everything
else neutral. Every figure on screen is read from poc_simulation/results/.

Run from the repository root:
    streamlit run dashboard/app.py
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RES_DIR = Path(__file__).resolve().parent.parent / "poc_simulation" / "results"

# =============================================================================
# Design tokens (single source of truth for CSS and charts)
# =============================================================================
TOKENS = {
    "paper": "#0E1217",     # canvas
    "paper-2": "#151B22",   # panels
    "paper-3": "#1C242D",   # raised panels / hover
    "rule": "#2A333E",      # hairlines, gridlines
    "ink": "#E7EBF0",       # primary text
    "muted": "#8B97A6",     # secondary text, axis labels
    "accent": "#3DBE8B",    # the proposed controller, safe band
    "warn": "#E3A33B",      # abnormal but expected (MSL lot open, guardrail acting)
    "alarm": "#E5544B",     # limit breached, fault detected
    "ref-1": "#5F6B7A",     # reference series: Static PID
    "ref-2": "#A3AEBB",     # reference series: SensorOnly MPC
}
FONT = "IBM Plex Sans"
FONT_STACK = "IBM Plex Sans, Helvetica Neue, Arial, sans-serif"  # charts need a fallback

TRACES = {
    "Static_PID": "sim_24h_A2_Static_PID.csv",
    "SensorOnly_MPC": "sim_24h_A2_SensorOnly_MPC.csv",
    "EquipState_MPC": "sim_24h_A2_EquipState_MPC.csv",
    "EquipState_MPC under 3 faults": "real_fault_guardrail_trace.csv",
}
LABELS = {
    "Static_PID": "Static PID (today's practice)",
    "SensorOnly_MPC": "Sensor-only MPC",
    "EquipState_MPC": "Equipment-state MPC (proposed)",
    "EquipState_MPC under 3 faults": "Equipment-state MPC under 3 injected faults",
}
SHORT = {"Static_PID": "Static PID", "SensorOnly_MPC": "Sensor-only MPC",
         "EquipState_MPC": "Equipment-state MPC"}
SERIES_COLOR = {
    "Static_PID": TOKENS["ref-1"],
    "SensorOnly_MPC": TOKENS["ref-2"],
    "EquipState_MPC": TOKENS["accent"],
}

T_LO, T_HI = 21.0, 23.0   # Tier-2 temperature window (report Table 1.2)
RH_LO = 40.0              # ESD floor

# (flag column, event name, guardrail response, colour)
EVENTS = [
    ("tier2_T_viol", "Zone temperature outside 21–23 °C", "Tier-2 excursion logged", "alarm"),
    ("tier2_RH_viol", "Zone humidity outside Tier-2 window", "Tier-2 excursion logged", "alarm"),
    ("tier1_viol", "Tier-1 hard limit breached", "Production hold", "alarm"),
    ("sensor_fault_detected", "Sensor 1 disagrees with sensors 2 and 3",
     "Sensor 1 isolated by 2-of-3 median vote", "alarm"),
    ("fallback_active", "SECS/GEM heartbeat lost or Tier-1 margin reached",
     "Bumpless transfer to SIMC PID, static setpoints", "warn"),
    ("valve_fault_active", "Cooling valve not tracking command",
     "CHW forced to 11.8 °C, FFU restored to 35 ACH", "alarm"),
    ("msl_ge3_active", "MSL ≥ 3 lot open", "RH ceiling tightened to 45 %", "warn"),
]


# =============================================================================
# Styling
# =============================================================================
def inject_css():
    root = "\n".join(f"  --color-{k}: {v};" for k, v in TOKENS.items())
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root {{
{root}
  --font-body: '{FONT}', ui-sans-serif, system-ui, sans-serif;
  --radius: 6px;
}}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, label {{
  font-family: var(--font-body) !important;
  font-variant-numeric: tabular-nums;
}}
.stApp {{ background: var(--color-paper); color: var(--color-ink); }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1440px; padding-top: 2.2rem; }}
h1, h2, h3 {{ color: var(--color-ink); font-weight: 600; letter-spacing: 0; }}
h1 {{ font-size: 1.6rem !important; margin-bottom: 0 !important; }}
h3 {{ font-size: 1.05rem !important; }}
.lede {{ color: var(--color-muted); font-size: 0.95rem; margin: 0.2rem 0 1.2rem; }}

.tile {{
  background: var(--color-paper-2); border: 1px solid var(--color-rule);
  border-radius: var(--radius); padding: 14px 16px; height: 100%;
}}
.tile .label {{ color: var(--color-muted); font-size: 0.85rem; }}
.tile .value {{ color: var(--color-ink); font-size: 1.9rem; font-weight: 500; line-height: 1.25; }}
.tile .note {{ color: var(--color-muted); font-size: 0.82rem; }}
.tile.alarm {{ border-color: var(--color-alarm); }}
.tile.alarm .note {{ color: var(--color-alarm); }}

.status {{
  display: flex; justify-content: space-between; gap: 12px; align-items: baseline;
  background: var(--color-paper-2); border: 1px solid var(--color-rule);
  border-left: 4px solid var(--color-rule); border-radius: var(--radius);
  padding: 9px 12px; margin-bottom: 6px;
}}
.status .name {{ color: var(--color-ink); font-size: 0.9rem; }}
.status .state {{ color: var(--color-muted); font-size: 0.85rem; text-align: right; }}
.status.warn  {{ border-left-color: var(--color-warn); }}
.status.warn .state  {{ color: var(--color-warn); }}
.status.alarm {{ border-left-color: var(--color-alarm); background: color-mix(in srgb, var(--color-alarm) 9%, var(--color-paper-2)); }}
.status.alarm .state {{ color: var(--color-alarm); font-weight: 600; }}

.section-label {{ color: var(--color-muted); font-size: 0.85rem; margin: 6px 0 6px; }}
table.log {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
table.log th {{ text-align: left; color: var(--color-muted); font-weight: 500;
  border-bottom: 1px solid var(--color-rule); padding: 8px 10px; }}
table.log td {{ color: var(--color-ink); border-bottom: 1px solid var(--color-rule); padding: 8px 10px; }}
table.log td.num {{ text-align: right; }}
table.log tr td:first-child {{ border-left: 3px solid transparent; }}
table.log tr.alarm td:first-child {{ border-left-color: var(--color-alarm); }}
table.log tr.warn td:first-child {{ border-left-color: var(--color-warn); }}
div[data-testid="stDataFrame"] {{ border: 1px solid var(--color-rule); border-radius: var(--radius); }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}
</style>
""",
        unsafe_allow_html=True,
    )


def styled(chart, height):
    """Apply the token palette to an Altair chart (hairline solid grid, muted axes, few ticks)."""
    return (
        chart.properties(height=height, background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=TOKENS["muted"], titleColor=TOKENS["muted"], labelFont=FONT_STACK, titleFont=FONT_STACK,
            labelFontSize=11, titleFontSize=11, titleFontWeight="normal",
            gridColor=TOKENS["rule"], gridWidth=1, domainColor=TOKENS["rule"],
            tickColor=TOKENS["rule"], tickCount=5,
        )
        .configure_legend(
            labelColor=TOKENS["ink"], titleColor=TOKENS["muted"], labelFont=FONT_STACK, titleFont=FONT_STACK,
            orient="top", direction="horizontal", title=None, labelFontSize=11, symbolStrokeWidth=3,
            labelLimit=0,
        )
        .configure_axisX(grid=False)
        .configure_title(color=TOKENS["ink"], font=FONT_STACK, fontSize=13, fontWeight=500, anchor="start")
    )


# =============================================================================
# Data
# =============================================================================
@st.cache_data
def load_trace(name: str) -> pd.DataFrame:
    df = pd.read_csv(RES_DIR / TRACES[name])
    df["clock"] = pd.to_datetime("2017-12-06") + pd.to_timedelta(df["minute"], unit="m")
    return df


def hhmm(minute: int) -> str:
    return f"{int(minute) // 60:02d}:{int(minute) % 60:02d}"


def event_log(df: pd.DataFrame, t: int) -> pd.DataFrame:
    """Contiguous runs of each flag up to minute t, newest first."""
    rows = []
    upto = df.iloc[: t + 1]
    for col, name, response, level in EVENTS:
        if col not in upto.columns:
            continue
        flag = upto[col].astype(int).values
        start = None
        for i, v in enumerate(list(flag) + [0]):
            if v and start is None:
                start = i
            elif not v and start is not None:
                ongoing = i == len(flag) and flag[-1] == 1
                rows.append({
                    "Start": hhmm(start),
                    "End": "ongoing" if ongoing else hhmm(i),
                    "Minutes": i - start,
                    "Event": name,
                    "Response": response,
                    "Level": level,
                    "_start": start,
                })
                start = None
    if not rows:
        return pd.DataFrame(columns=["Start", "End", "Minutes", "Event", "Response", "Level"])
    out = pd.DataFrame(rows).sort_values("_start", ascending=False).drop(columns="_start")
    return out.reset_index(drop=True)


# =============================================================================
# Components
# =============================================================================
def tile(label, value, note, alarm=False):
    cls = "tile alarm" if alarm else "tile"
    return (f"<div class='{cls}'><div class='label'>{label}</div>"
            f"<div class='value'>{value}</div><div class='note'>{note}</div></div>")


def status_row(name, state_text, level="normal"):
    cls = "status" if level == "normal" else f"status {level}"
    return (f"<div class='{cls}'><span class='name'>{name}</span>"
            f"<span class='state'>{state_text}</span></div>")


def log_table(log: pd.DataFrame) -> str:
    """Event log as a plain HTML table: every row visible, severity shown by edge colour and by the words."""
    head = "".join(f"<th>{c}</th>" for c in ["Start", "End", "Minutes", "Event", "Response"])
    body = "".join(
        f"<tr class='{r.Level}'><td>{r.Start}</td><td>{r.End}</td><td class='num'>{r.Minutes}</td>"
        f"<td>{r.Event}</td><td>{r.Response}</td></tr>"
        for r in log.itertuples()
    )
    return f"<table class='log'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def envelope_chart(df, y, title, unit, lo, hi, domain, color, extra=None):
    """The signature view: a trace inside its Tier-2 safety envelope."""
    band_df = df[["clock"]].assign(lo=lo, hi=hi if not isinstance(hi, str) else df[hi])
    band = alt.Chart(band_df).mark_area(color=TOKENS["accent"], opacity=0.08).encode(
        x="clock:T", y=alt.Y("lo:Q", scale=alt.Scale(domain=domain)), y2="hi:Q")
    limits = alt.Chart(band_df).mark_line(color=TOKENS["alarm"], strokeWidth=1, opacity=0.7)
    hi_line = limits.encode(x="clock:T", y="hi:Q")
    lo_line = limits.encode(x="clock:T", y="lo:Q")
    line = alt.Chart(df).mark_line(color=color, strokeWidth=2).encode(
        x=alt.X("clock:T", title=None, axis=alt.Axis(format="%H:%M")),
        y=alt.Y(f"{y}:Q", title=unit, scale=alt.Scale(domain=domain)),
        tooltip=[alt.Tooltip("clock:T", title="Time", format="%H:%M"),
                 alt.Tooltip(f"{y}:Q", title=title, format=".2f")],
    )
    layers = [band, hi_line, lo_line, line] + ([extra] if extra is not None else [])
    return alt.layer(*layers).properties(title=title)


# =============================================================================
# Page
# =============================================================================
st.set_page_config(page_title="Cleanroom HVAC operator view", layout="wide")
inject_css()

st.markdown("# Cleanroom HVAC operator view")
st.markdown(
    "<p class='lede'>ISO 7 packaging bay of 1,500 m², replayed minute by minute on the held-out "
    "peak production day of 6 December 2017 (real HIPE equipment load, Scenario A2).</p>",
    unsafe_allow_html=True,
)

missing = [f for f in TRACES.values() if not (RES_DIR / f).exists()]
if missing:
    st.error(
        "Simulation results are missing: " + ", ".join(missing)
        + ". Run `python poc_simulation/run_poc_real.py` from the repository root, then reload."
    )
    st.stop()

VIEWS = ["Live operation", "PLC fault test", "Controller comparison"]
seg = getattr(st, "segmented_control", None)
view = (seg("View", VIEWS, default=VIEWS[0], label_visibility="collapsed") if seg
        else st.radio("View", VIEWS, horizontal=True, label_visibility="collapsed")) or VIEWS[0]

# -----------------------------------------------------------------------------
# Views 1 and 2: replay one trace
# -----------------------------------------------------------------------------
if view in ("Live operation", "PLC fault test"):
    if view == "Live operation":
        c_sel, c_time = st.columns([1, 3])
        trace_name = c_sel.selectbox(
            "Controller", ["EquipState_MPC", "SensorOnly_MPC", "Static_PID"],
            format_func=lambda k: LABELS[k])
        default_t = 615
    else:
        trace_name = "EquipState_MPC under 3 faults"
        c_sel, c_time = st.columns([1, 3])
        c_sel.markdown(
            "<div class='section-label'>Injected faults, 60 min each: Sensor 1 drift +1.8 °C at 09:30, "
            "SECS/GEM link loss at 13:00, cooling valve stuck at 78 % at 15:30.</div>",
            unsafe_allow_html=True)
        default_t = 945

    df = load_trace(trace_name)
    clock_opts = [hhmm(m) for m in range(len(df))]
    t = clock_opts.index(c_time.select_slider("Replay time", options=clock_opts, value=hhmm(default_t)))
    now, upto = df.iloc[t], df.iloc[: t + 1]
    pid = load_trace("Static_PID")
    color = TOKENS["accent"] if "EquipState" in trace_name else SERIES_COLOR[trace_name]

    # --- Stat row: four figures, each with a worded qualifier ---
    t_margin = T_HI - now["T_z"]
    t_out = not (T_LO <= now["T_z"] <= T_HI)
    rh_hi = now["RH_tier2_hi"]
    rh_out = not (RH_LO <= now["RH_z"] <= rh_hi)
    save = 100 * (pid["P_hvac_total_kW"].iloc[: t + 1].mean() - upto["P_hvac_total_kW"].mean()) \
        / pid["P_hvac_total_kW"].iloc[: t + 1].mean()
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(tile("Zone temperature", f"{now['T_z']:.2f} °C",
                     f"{abs(t_margin):.2f} K {'above' if t_margin < 0 else 'below'} the 23 °C limit"
                     if now["T_z"] >= 22 else f"{now['T_z'] - T_LO:.2f} K above the 21 °C limit",
                     alarm=t_out), unsafe_allow_html=True)
    k2.markdown(tile("Relative humidity", f"{now['RH_z']:.1f} %",
                     f"Window 40–{rh_hi:.0f} % {'(MSL ≥ 3 lot open)' if now['msl_ge3_active'] else ''}",
                     alarm=rh_out), unsafe_allow_html=True)
    k3.markdown(tile("HVAC electric power", f"{now['P_hvac_total_kW']:.0f} kW",
                     f"Static PID draws {pid['P_hvac_total_kW'].iloc[t]:.0f} kW at this minute"),
                unsafe_allow_html=True)
    k4.markdown(tile("Energy saved since 00:00", f"{save:.1f} %",
                     "Against Static PID on the same day"), unsafe_allow_html=True)
    st.write("")

    # --- Status column + envelope charts ---
    left, right = st.columns([1, 2.6], gap="medium")
    with left:
        st.markdown("<div class='section-label'>Bay status</div>", unsafe_allow_html=True)
        rows = [
            status_row("Temperature", "Outside 21–23 °C" if t_out else "Normal",
                       "alarm" if t_out else "normal"),
            status_row("Humidity", f"Outside 40–{rh_hi:.0f} %" if rh_out else "Normal",
                       "alarm" if rh_out else "normal"),
            status_row("Moisture-sensitive lot", "Open, RH ceiling 45 %" if now["msl_ge3_active"]
                       else "None open", "warn" if now["msl_ge3_active"] else "normal"),
        ]
        if "EquipState" in trace_name or "SensorOnly" in trace_name:
            rows += [
                status_row("Zone sensors", "Sensor 1 isolated" if now["sensor_fault_detected"]
                           else "3 of 3 agree", "alarm" if now["sensor_fault_detected"] else "normal"),
                status_row("Control mode", "PID fallback" if now["fallback_active"] else "MPC",
                           "warn" if now["fallback_active"] else "normal"),
                status_row("Cooling valve", "Stuck, interlock on" if now["valve_fault_active"]
                           else "Tracking", "alarm" if now["valve_fault_active"] else "normal"),
            ]
        st.markdown("".join(rows), unsafe_allow_html=True)
        st.markdown(
            f"<div class='section-label' style='margin-top:14px'>Minutes outside Tier-2 so far: "
            f"temperature {int(upto['tier2_T_viol'].sum())}, humidity {int(upto['tier2_RH_viol'].sum())}. "
            f"Tier-1: {int(upto['tier1_viol'].sum())}.</div>",
            unsafe_allow_html=True)

    with right:
        extra = None
        if view == "PLC fault test":
            # Draw the raw Sensor 1 reading only while the vote has isolated it
            raw = upto.assign(T_s1_raw=upto["T_s1_raw"].where(upto["sensor_fault_detected"] == 1))
            extra = alt.Chart(raw).mark_line(color=TOKENS["warn"], strokeWidth=1.5).encode(
                x="clock:T", y=alt.Y("T_s1_raw:Q", scale=alt.Scale(domain=[20.5, 25.0])),
                tooltip=[alt.Tooltip("T_s1_raw:Q", title="Sensor 1 raw", format=".2f")])
        st.altair_chart(styled(envelope_chart(
            upto, "T_z", "Zone temperature inside the 21–23 °C envelope", "°C",
            T_LO, T_HI, [20.5, 25.0] if view == "PLC fault test" else [20.5, 24.2], color, extra), 250), width="stretch", theme=None)
        if view == "PLC fault test":
            st.markdown("<div class='section-label'>Amber line: raw Sensor 1 reading while the 2-of-3 vote "
                        "ignores it. Green line: true zone temperature.</div>", unsafe_allow_html=True)
        st.altair_chart(styled(envelope_chart(
            upto, "RH_z", "Relative humidity inside the ESD / MSL envelope", "% RH",
            RH_LO, "RH_tier2_hi", [38, 52], color), 210), width="stretch", theme=None)

    # --- Event log: every abnormal state, derived from the trace flags ---
    st.markdown("### Event log")
    log = event_log(df, t)
    if log.empty:
        st.markdown("<div class='section-label'>No abnormal events up to this minute.</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(log_table(log), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# View 3: controller comparison
# -----------------------------------------------------------------------------
else:
    names = ["Static_PID", "SensorOnly_MPC", "EquipState_MPC"]
    frames = {n: load_trace(n) for n in names}
    base = frames["Static_PID"]["P_hvac_total_kW"].mean()

    eq = frames["EquipState_MPC"]
    eq_save = 100 * (base - eq["P_hvac_total_kW"].mean()) / base
    k1, k2, k3 = st.columns(3)
    k1.markdown(tile("Electricity saved, proposed controller", f"{eq_save:.2f} %",
                     f"{(base - eq['P_hvac_total_kW'].mean()) * 8.76:.0f} MWh per year against Static PID"),
                unsafe_allow_html=True)
    k2.markdown(tile("Minutes outside Tier-2, proposed controller",
                     f"{int(eq['tier2_viol'].sum())}",
                     f"Static PID {int(frames['Static_PID']['tier2_viol'].sum())}, "
                     f"sensor-only MPC {int(frames['SensorOnly_MPC']['tier2_viol'].sum())}"),
                unsafe_allow_html=True)
    k3.markdown(tile("Peak zone temperature, proposed controller", f"{eq['T_z'].max():.2f} °C",
                     f"{T_HI - eq['T_z'].max():.2f} K below the 23 °C limit"), unsafe_allow_html=True)
    st.write("")

    st.markdown("### The 10:12 ChipPress surge")
    zoom = st.slider("Time window (hours)", 0.0, 24.0, (9.5, 11.5), step=0.25)
    long = pd.concat([d.assign(Controller=LABELS[n]) for n, d in frames.items()])
    long = long[(long["hour"] >= zoom[0]) & (long["hour"] <= zoom[1])]
    order = [LABELS[n] for n in names]
    colour = alt.Color("Controller:N", sort=order,
                       scale=alt.Scale(domain=order, range=[SERIES_COLOR[n] for n in names]))
    limit = alt.Chart(pd.DataFrame({"y": [T_HI]})).mark_rule(strokeWidth=1.5).encode(
        y="y:Q", color=alt.value(TOKENS["alarm"]))
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("clock:T", title=None, axis=alt.Axis(format="%H:%M")),
        y=alt.Y("T_z:Q", title="°C", scale=alt.Scale(zero=False)),
        color=colour,
        tooltip=["Controller", alt.Tooltip("clock:T", title="Time", format="%H:%M"),
                 alt.Tooltip("T_z:Q", title="Zone °C", format=".2f")],
    )
    st.altair_chart(styled(alt.layer(lines, limit).properties(
        title="Zone temperature against the 23 °C Tier-2 limit (red line)"), 300),
        width="stretch", theme=None)

    left, right = st.columns([1.3, 1], gap="medium")
    with left:
        st.markdown("### 24-hour summary")
        rows = []
        for n, d in frames.items():
            kw = d["P_hvac_total_kW"].mean()
            rows.append({
                "Controller": LABELS[n],
                "Energy (MWh/yr)": round(kw * 8.76, 1),
                "Saved vs PID (%)": round(100 * (base - kw) / base, 2),
                "Peak temp (°C)": round(d["T_z"].max(), 2),
                "Tier-2 min, T / RH":
                    f"{int(d['tier2_T_viol'].sum())} / {int(d['tier2_RH_viol'].sum())}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown(
            "<div class='section-label'>Sensor-only MPC saves 0.19 points more electricity because it "
            "does not pre-chill before the surge, and it crosses 23 °C for 3 minutes. Only the proposed "
            "controller passes the zero-excursion gate.</div>", unsafe_allow_html=True)
    with right:
        st.markdown("### Where the electricity goes")
        parts = []
        for n, d in frames.items():
            for col, part in [("P_chiller_kW", "Chillers"), ("P_reheat_kW", "Electric reheat"),
                              ("P_fans_kW", "Fans")]:
                parts.append({"Controller": SHORT[n], "Part": part, "MWh/yr": d[col].mean() * 8.76})
        bars = alt.Chart(pd.DataFrame(parts)).mark_bar(size=22).encode(
            y=alt.Y("Controller:N", sort=[SHORT[n] for n in names], title=None),
            x=alt.X("sum(MWh/yr):Q", title="MWh per year", stack=True),
            color=alt.Color("Part:N", sort=["Chillers", "Electric reheat", "Fans"],
                            scale=alt.Scale(range=["#3DBE8B", "#2B8A65", "#1D5C45"])),
            order=alt.Order("Part:N"),
            tooltip=["Controller", "Part", alt.Tooltip("MWh/yr:Q", format=".0f")],
        )
        st.altair_chart(styled(bars, 190), width="stretch", theme=None)
