"""PitMind Dashboard — Streamlit UI for ACC telemetry analysis.

Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# PitMind modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pitmind.config import Config
from pitmind import features, mistakes, timeloss, potential_lap, coaching, segmentation, corners
from synthetic import generator as gen
from dashboard import map_plot


st.set_page_config(
    page_title="PitMind — AI Driver Coach",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_session(laps: int = 12, track: str = "monza") -> tuple[pd.DataFrame, dict]:
    """Generate or load session data (cached)."""
    session, gt = gen.generate_session(laps=laps, track_kind=track)
    return session, gt


@st.cache_data
def process_session(_session: pd.DataFrame, cfg: Config) -> dict:
    """Run full pipeline on session (cached)."""
    table = features.build_feature_table(_session, cfg)
    mistake_list = mistakes.detect_mistakes(table, cfg)
    time_loss_list = timeloss.estimate_time_loss(mistake_list, table, cfg)
    pot = potential_lap.build_potential_lap(_session, cfg)
    directives = coaching.generate_directives(mistake_list, time_loss_list)
    totals = timeloss.total_time_loss_per_lap(time_loss_list)
    return {
        "table": table,
        "mistakes": mistake_list,
        "time_losses": time_loss_list,
        "potential_lap": pot,
        "directives": directives,
        "totals": totals,
    }


# Sidebar
st.sidebar.title("🏎️ PitMind")
st.sidebar.caption("AI Driver Coach for ACC")

track_options = ["monza", "spa", "silverstone", "imola", "generic"]
track = st.sidebar.selectbox("Circuit", track_options, index=0)
laps = st.sidebar.slider("Laps", 4, 20, 12)
st.sidebar.divider()

# Generate/load data
with st.spinner(f"Generating {laps} laps on {track.title()}..."):
    session, gt = load_session(laps, track)
    cfg = Config.from_file()
    results = process_session(session, cfg)

table = results["table"]
mistake_list = results["mistakes"]
time_loss_list = results["time_losses"]
pot = results["potential_lap"]
directives = results["directives"]
totals = results["totals"]

# Header
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Circuit", track.title())
with col2:
    st.metric("Valid Laps", session["lap_number"].nunique())
with col3:
    st.metric("Potential Lap", f"{pot.total_time_s:.3f}s", 
              delta=f"{pot.improvement_vs_best_s:+.3f}s vs best")

st.divider()

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Telemetry", 
    "🎯 Corners", 
    "⚠️ Mistakes", 
    "🏁 Potential Lap", 
    "📋 Coaching",
    "🗺️ Track Map"
])

# ---- TAB 1: Telemetry ----
with tab1:
    st.subheader("Telemetry Overlay")
    
    lap_numbers = sorted(session["lap_number"].unique())
    selected_laps = st.multiselect("Select laps to overlay", lap_numbers, default=lap_numbers[:3])
    
    # Channel selector
    channels = st.multiselect(
        "Channels", 
        ["speed_kmh", "throttle", "brake", "steering", "gear", "rpm"],
        default=["speed_kmh", "throttle", "brake"]
    )
    
    if selected_laps and channels:
        fig = make_subplots(
            rows=len(channels), cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            subplot_titles=channels
        )
        
        for lap_num in selected_laps:
            lap_df = session[session["lap_number"] == lap_num].sort_values("track_position")
            tp = lap_df["track_position"]
            
            for i, ch in enumerate(channels):
                fig.add_trace(
                    go.Scatter(x=tp, y=lap_df[ch], name=f"L{lap_num}", 
                               mode="lines", line=dict(width=1.5)),
                    row=i+1, col=1
                )
        
        fig.update_layout(height=200*len(channels), showlegend=True, 
                          xaxis_title="Track Position")
        st.plotly_chart(fig, use_container_width=True)

# ---- TAB 2: Corners ----
with tab2:
    st.subheader("Corner Analysis Table")
    
    # Lap selector
    lap_nums = sorted(table["lap"].unique())
    sel_lap = st.selectbox("Select lap", lap_nums, index=0)
    
    lap_table = table[table["lap"] == sel_lap].copy()
    
    # Format for display
    display_cols = ["corner", "name", "entry_speed_kmh", "apex_speed_kmh", 
                    "exit_speed_kmh", "brake_point_m", "throttle_on_s", 
                    "corner_time_s", "delta_apex_speed_kmh", "delta_brake_point_m",
                    "delta_throttle_on_s", "delta_corner_time_s"]
    
    display_df = lap_table[display_cols].round(2)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Corner time comparison chart
    st.subheader("Corner Time vs Reference")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lap_table["name"], 
        y=lap_table["corner_time_s"],
        name=f"Lap {sel_lap}",
        marker_color="steelblue"
    ))
    # Reference (lap 1)
    ref_table = table[table["lap"] == 1].sort_values("corner")
    fig.add_trace(go.Bar(
        x=ref_table["name"], 
        y=ref_table["corner_time_s"],
        name="Reference (L1)",
        marker_color="lightgray"
    ))
    fig.update_layout(barmode="group", yaxis_title="Corner Time (s)", xaxis_title="Corner")
    st.plotly_chart(fig, use_container_width=True)

# ---- TAB 3: Mistakes ----
with tab3:
    st.subheader("Detected Mistakes")
    
    # Convert to DataFrame
    m_df = mistakes.mistakes_to_dataframe(mistake_list)
    tl_df = timeloss.time_loss_to_dataframe(time_loss_list)
    
    if not m_df.empty:
        # Merge with time loss
        merged = m_df.merge(
            tl_df[["lap", "corner", "mistake_type", "time_loss_s"]],
            on=["lap", "corner", "mistake_type"], how="left"
        )
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            lap_filter = st.multiselect("Filter by lap", sorted(merged["lap"].unique()), 
                                        default=sorted(merged["lap"].unique()))
        with col2:
            type_filter = st.multiselect("Filter by type", sorted(merged["mistake_type"].unique()),
                                         default=sorted(merged["mistake_type"].unique()))
        with col3:
            conf_filter = st.multiselect("Filter by confidence", sorted(merged["confidence"].unique()),
                                         default=sorted(merged["confidence"].unique()))
        
        filtered = merged[
            merged["lap"].isin(lap_filter) & 
            merged["mistake_type"].isin(type_filter) &
            merged["confidence"].isin(conf_filter)
        ].sort_values(["lap", "corner", "time_loss_s"], ascending=[True, True, False])
        
        st.dataframe(
            filtered[["lap", "corner_name", "mistake_type", "confidence", 
                      "delta_value", "threshold_value", "time_loss_s", "message"]].round(3),
            use_container_width=True, hide_index=True
        )
        
        # Summary charts
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Mistakes by Type")
            type_counts = filtered["mistake_type"].value_counts()
            fig = go.Figure(go.Bar(x=type_counts.index, y=type_counts.values))
            fig.update_layout(yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Time Loss by Lap")
            lap_loss = pd.DataFrame(list(totals.items()), columns=["lap", "loss_s"])
            fig = go.Figure(go.Bar(x=lap_loss["lap"], y=lap_loss["loss_s"], marker_color="crimson"))
            fig.update_layout(yaxis_title="Time Loss (s)", xaxis_title="Lap")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No mistakes detected!")

# ---- TAB 4: Potential Lap ----
with tab4:
    st.subheader("Potential (Best-Sector) Lap")
    
    pot_df = potential_lap.potential_lap_to_dataframe(pot)
    st.dataframe(pot_df.round(3), use_container_width=True, hide_index=True)
    
    # Sector source visualization
    st.subheader("Sector Sources")
    sector_info = []
    for st_sec in pot.sector_times:
        sector_info.append({
            "Sector": f"S{st_sec.sector}",
            "Time (s)": f"{st_sec.time_s:.3f}",
            "Source Lap": f"L{st_sec.lap}",
        })
    st.table(pd.DataFrame(sector_info))
    
    # Potential lap telemetry overlay
    st.subheader("Potential Lap vs Best Actual")
    try:
        pot_telemetry = potential_lap.interpolate_potential_telemetry(pot, session, cfg)
        
        best_lap = min(lap_nums, key=lambda l: 
            session[session.lap_number==l]["timestamp"].max() - 
            session[session.lap_number==l]["timestamp"].min())
        best_df = session[session.lap_number == best_lap].sort_values("track_position")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           subplot_titles=["Speed", "Throttle/Brake"])
        
        fig.add_trace(go.Scatter(x=pot_telemetry["track_position"], y=pot_telemetry["speed_kmh"],
                                 name="Potential", line=dict(color="gold", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=best_df["track_position"], y=best_df["speed_kmh"],
                                 name=f"Best Actual (L{best_lap})", line=dict(color="steelblue", width=1.5)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=pot_telemetry["track_position"], y=pot_telemetry["throttle"],
                                 name="Potential Throttle", line=dict(color="green", width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=pot_telemetry["track_position"], y=pot_telemetry["brake"],
                                 name="Potential Brake", line=dict(color="red", width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=best_df["track_position"], y=best_df["throttle"],
                                 name=f"Actual Throttle", line=dict(color="lightgreen", width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=best_df["track_position"], y=best_df["brake"],
                                 name=f"Actual Brake", line=dict(color="pink", width=1)), row=2, col=1)
        
        fig.update_layout(height=500, xaxis_title="Track Position")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate potential telemetry: {e}")

# ---- TAB 5: Coaching ----
with tab5:
    st.subheader("Coaching Directives")
    
    if directives:
        dir_df = coaching.directives_to_dataframe(directives)
        
        # Priority filter
        prio_filter = st.multiselect("Priority", [1, 2, 3], default=[1, 2])
        filtered_dir = dir_df[dir_df["priority"].isin(prio_filter)]
        
        for _, row in filtered_dir.iterrows():
            priority_colors = {1: "🔴", 2: "🟡", 3: "🟢"}
            badge = priority_colors.get(row["priority"], "⚪")
            
            with st.container():
                col1, col2 = st.columns([1, 6])
                with col1:
                    st.markdown(f"**{badge} P{row['priority']}**")
                    st.caption(f"L{row['lap']} {row['corner']}")
                with col2:
                    st.markdown(f"**{row['message']}**")
                    st.caption(f"Category: {row['category']} | Confidence: {row['confidence']} | Est. loss: {row['time_loss_s']:.2f}s")
        
        # Full report
        st.divider()
        with st.expander("📄 Full Session Report"):
            report = coaching.generate_session_report(directives, pot, totals)
            st.code(report, language="text")
    else:
        st.success("No coaching directives needed — great driving!")

# ---- TAB 6: Track Map ----
with tab6:
    st.subheader("Circuit Track Map")

    lap1_df = session[session["lap_number"] == 1].reset_index(drop=True)
    if len(lap1_df) == 0:
        first_lap = int(session["lap_number"].min())
        lap1_df = session[session["lap_number"] == first_lap].reset_index(drop=True)
    detected_corners = corners.detect_corners(lap1_df, cfg)

    map_laps = sorted(session["lap_number"].unique())
    map_lap = st.selectbox("Select lap to map", map_laps, index=len(map_laps) - 1,
                           key="track_map_lap")
    color_by_speed = st.checkbox("Color by speed (heat-map)", value=True,
                                 key="track_map_speed")

    map_df = session[session["lap_number"] == map_lap].sort_values("track_position")
    # centre on lap 1 so every lap shares the same origin
    fig_map = map_plot.track_map_figure(
        map_df,
        detected_corners,
        color_by_speed=color_by_speed,
        title=f"Lap {map_lap} — {track.title()}",
        ref=lap1_df,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.caption(
        "Track shape is drawn from the driven x/y path (never a GeoJSON), so the same "
        "view works for recorded ACC laps too. Red dots = detected corner apexes; "
        "green square = start/finish line; ribbon color = speed."
    )

# Footer
st.divider()
st.caption("PitMind v0.1 — AI Driver Coach for ACC | Built with Streamlit & Plotly")