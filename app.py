import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html
from plotly.subplots import make_subplots


TEAL = "#8BCAD3"
TITLE_TEAL = "#8FCED8"
CORAL = "#F28378"
TEXT = "#A9A9A9"
SUBTEXT = "#B5B5B5"
AXIS = "#B8B8B8"
HEX_BORDER = "#D9D9D9"
WHITE = "#FFFFFF"

REQUIRED_COLUMNS = {
    "Complaint ID",
    "Product",
    "Issue",
    "State",
    "Date received",
    "Company",
    "Company response",
}

VALID_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
}

# Book-style US tile map positions. Plotly shapes below render flat-topped hexagons.
HEX_COORDS = {
    "AK": (0.0, 0), "ME": (11.0, 0),
    "VT": (9.5, 1), "NH": (10.5, 1),
    "WA": (1.0, 2), "MT": (2.0, 2), "ND": (3.0, 2), "MN": (4.0, 2),
    "WI": (5.0, 2), "MI": (7.0, 2), "NY": (8.5, 2), "MA": (9.5, 2),
    "RI": (10.5, 2),
    "ID": (1.5, 3), "WY": (2.5, 3), "SD": (3.5, 3), "IA": (4.5, 3),
    "IL": (5.5, 3), "IN": (6.5, 3), "OH": (7.5, 3), "PA": (8.5, 3),
    "NJ": (9.5, 3), "CT": (10.5, 3),
    "OR": (1.0, 4), "NV": (2.0, 4), "CO": (3.0, 4), "NE": (4.0, 4),
    "MO": (5.0, 4), "KY": (6.0, 4), "WV": (7.0, 4), "MD": (8.0, 4),
    "DE": (9.0, 4),
    "CA": (1.5, 5), "AZ": (2.5, 5), "UT": (3.5, 5), "KS": (4.5, 5),
    "AR": (5.5, 5), "TN": (6.5, 5), "VA": (7.5, 5), "NC": (8.5, 5),
    "DC": (9.5, 5),
    "NM": (3.0, 6), "OK": (4.0, 6), "LA": (5.0, 6), "MS": (6.0, 6),
    "AL": (7.0, 6), "SC": (8.0, 6),
    "TX": (3.5, 7), "GA": (7.5, 7),
    "HI": (0.0, 8), "FL": (8.0, 8),
}


def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path}. Place 26k-consumer-complaints.csv next to app.py "
            "or set the COMPLAINTS_CSV environment variable."
        )
    return pd.read_csv(path, low_memory=False)


def prepare_data(df):
    df = df.drop(columns=["Unnamed: 0"], errors="ignore").copy()
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError("The CSV is missing required columns: " + ", ".join(missing))

    df["Date received"] = pd.to_datetime(df["Date received"], errors="coerce")
    df = df.loc[df["Date received"].notna()].copy()
    state = df["State"].fillna("").astype(str).str.upper().str.strip()
    df["State"] = state.where(state.isin(VALID_STATES), pd.NA)
    df["Status"] = np.where(df["Company response"].eq("In progress"), "Open", "Closed")
    df["Month"] = df["Date received"].dt.to_period("M").dt.to_timestamp()
    df["Reason"] = df["Issue"].fillna("Unknown").astype(str)
    df["Source Type"] = df["Product"].fillna("Unknown").astype(str)
    # The source dataset has no Party field, so Company is used as its proxy.
    df["PartyRaw"] = df["Company"].fillna("Unknown").astype(str)
    return df


def apply_filters(
    df,
    start_date,
    end_date,
    source_type="All",
    status="All",
    selected_state=None,
    selected_reason=None,
    selected_party=None,
):
    mask = df["Date received"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    if source_type != "All":
        mask &= df["Source Type"].eq(source_type)
    if status != "All":
        mask &= df["Status"].eq(status)
    if selected_state:
        mask &= df["State"].eq(selected_state)
    if selected_reason:
        mask &= df["Reason"].eq(selected_reason)
    if selected_party and selected_party.get("members"):
        mask &= df["PartyRaw"].isin(selected_party["members"])
    return df.loc[mask].copy()


def chart_layout(**kwargs):
    layout = dict(
        height=270,
        autosize=True,
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(family="Montserrat, Avenir Next, Segoe UI, Arial, sans-serif", color=TEXT),
        showlegend=False,
        margin=dict(l=34, r=18, t=28, b=32),
        hoverlabel=dict(
            bgcolor=WHITE,
            bordercolor=HEX_BORDER,
            font=dict(
                family="Montserrat, Avenir Next, Segoe UI, Arial, sans-serif",
                size=10,
                color=TEXT,
            ),
        ),
    )
    layout.update(kwargs)
    return layout


def make_kpi(df):
    counts = df["Status"].value_counts()
    return int(counts.get("Closed", 0)), int(counts.get("Open", 0)), int(len(df))


def make_month_chart(df, global_min, global_max):
    months = pd.date_range(global_min.to_period("M").start_time, global_max.to_period("M").start_time, freq="MS")
    grouped = (
        df.groupby(["Month", "Status"]).size().unstack(fill_value=0)
        .reindex(index=months, columns=["Open", "Closed"], fill_value=0)
    )
    fig = go.Figure()
    for status, color in [("Open", CORAL), ("Closed", TEAL)]:
        fig.add_bar(
            x=months,
            y=grouped[status],
            name=status,
            marker_color=color,
            marker_line_width=0,
            hovertemplate=f"%{{x|%b %Y}}<br>{status}: %{{y:,}}<extra></extra>",
        )

    year_annotations = []
    for year in sorted(set(months.year)):
        year_months = months[months.year == year]
        year_annotations.append(
            dict(
                x=year_months[0],
                y=1.08,
                xref="x",
                yref="paper",
                text=str(year),
                showarrow=False,
                xanchor="center",
                font=dict(size=11, color=TEXT),
            )
        )

    fig.update_layout(
        **chart_layout(margin=dict(l=34, r=18, t=46, b=34)),
        barmode="stack",
        bargap=0.13,
        annotations=year_annotations,
        xaxis=dict(
            tickmode="array",
            tickvals=months,
            ticktext=[m.strftime("%b") for m in months],
            tickfont=dict(size=10, color=TEXT),
            showgrid=False,
            showline=True,
            linecolor=AXIS,
            fixedrange=True,
        ),
        yaxis=dict(
            rangemode="tozero",
            tickfont=dict(size=10, color=TEXT),
            gridcolor="#EEEEEE",
            gridwidth=0.7,
            zeroline=False,
            fixedrange=True,
        ),
    )
    return fig


def _positive_ticks(maximum):
    maximum = max(1, int(maximum))
    step = max(1, int(np.ceil(maximum / 3)))
    bound = step * 3
    vals = list(range(0, bound + 1, step))
    return vals, [f"{v:,}" for v in vals], bound


def _make_split_bar_chart(categories, closed, opened, closed_custom, open_custom, selected_label=None):
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.07,
        column_widths=[0.5, 0.5],
    )
    closed_colors = ["#64B9C5" if value == selected_label else TEAL for value in categories]
    open_colors = ["#EC6E63" if value == selected_label else CORAL for value in categories]
    closed_hover = [f"{label}<br>Closed: {value:,}" for label, value in zip(categories, closed)]
    open_hover = [f"{label}<br>Open: {value:,}" for label, value in zip(categories, opened)]

    fig.add_bar(
        x=closed,
        y=categories,
        orientation="h",
        marker=dict(color=closed_colors, line_width=0),
        customdata=closed_custom,
        hovertext=closed_hover,
        hovertemplate="%{hovertext}<extra></extra>",
        row=1,
        col=1,
    )
    fig.add_bar(
        x=opened,
        y=categories,
        orientation="h",
        marker=dict(color=open_colors, line_width=0),
        customdata=open_custom,
        hovertext=open_hover,
        hovertemplate="%{hovertext}<extra></extra>",
        row=1,
        col=2,
    )

    # Closed and Open use the same scale within each chart so bar lengths are directly comparable.
    common_ticks, common_ticktext, common_bound = _positive_ticks(
        max(max(closed, default=0), max(opened, default=0))
    )
    fig.update_layout(
        **chart_layout(margin=dict(l=202, r=18, t=36, b=32)),
        bargap=0.38,
        annotations=[
            dict(x=0, y=1.09, xref="x domain", yref="paper", xanchor="left", text="Closed", showarrow=False, font=dict(size=11, color=TEXT)),
            dict(x=0, y=1.09, xref="x2 domain", yref="paper", xanchor="left", text="Open", showarrow=False, font=dict(size=11, color=TEXT)),
        ],
    )
    fig.update_xaxes(
        range=[0, common_bound],
        tickvals=common_ticks,
        ticktext=common_ticktext,
        tickfont=dict(size=9, color=SUBTEXT),
        showgrid=False,
        zeroline=True,
        zerolinecolor=AXIS,
        zerolinewidth=1,
        showline=True,
        linecolor=AXIS,
        fixedrange=True,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        range=[0, common_bound],
        tickvals=common_ticks,
        ticktext=common_ticktext,
        tickfont=dict(size=9, color=SUBTEXT),
        showgrid=False,
        zeroline=True,
        zerolinecolor=AXIS,
        zerolinewidth=1,
        showline=True,
        linecolor=AXIS,
        fixedrange=True,
        row=1,
        col=2,
    )
    fig.update_yaxes(
        autorange="reversed",
        tickfont=dict(size=9, color=TEXT),
        fixedrange=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(showticklabels=False, fixedrange=True, row=1, col=2)
    return fig


def make_reason_chart(df, selected_reason=None):
    grouped = df.groupby(["Reason", "Status"]).size().unstack(fill_value=0)
    for col in ["Open", "Closed"]:
        if col not in grouped:
            grouped[col] = 0
    grouped["Total"] = grouped["Open"] + grouped["Closed"]
    grouped = grouped.sort_values("Total", ascending=False).head(12)
    reasons = grouped.index.tolist()
    return _make_split_bar_chart(
        reasons,
        grouped["Closed"].tolist(),
        grouped["Open"].tolist(),
        reasons,
        reasons,
        selected_reason,
    )


def party_groups(df, top_n=8):
    top = df["PartyRaw"].value_counts().head(top_n).index.tolist()
    grouped = df.assign(Party=np.where(df["PartyRaw"].isin(top), df["PartyRaw"], "Other"))
    members = {name: [name] for name in top}
    members["Other"] = sorted(df.loc[~df["PartyRaw"].isin(top), "PartyRaw"].unique().tolist())
    return grouped, members


def make_party_chart(df, selected_party=None):
    party_df, members = party_groups(df)
    grouped = party_df.groupby(["Party", "Status"]).size().unstack(fill_value=0)
    for col in ["Open", "Closed"]:
        if col not in grouped:
            grouped[col] = 0
    grouped["Total"] = grouped["Open"] + grouped["Closed"]
    grouped = grouped.sort_values("Total", ascending=False)
    parties = grouped.index.tolist()
    selected_label = selected_party.get("label") if selected_party else None
    custom = [json.dumps({"label": p, "members": members.get(p, [])}) for p in parties]
    fig = _make_split_bar_chart(
        parties,
        grouped["Closed"].tolist(),
        grouped["Open"].tolist(),
        custom,
        custom,
        selected_label,
    )
    fig.update_layout(margin=dict(l=168, r=18, t=36, b=32), bargap=0.35)
    return fig


def _hex_path(x, y, radius=0.43):
    # Pointy-topped hexagon: sharp vertices at the top and bottom, as in the book.
    half_width = np.sqrt(3) * radius / 2
    points = [
        (x, y - radius),
        (x + half_width, y - radius / 2),
        (x + half_width, y + radius / 2),
        (x, y + radius),
        (x - half_width, y + radius / 2),
        (x - half_width, y - radius / 2),
    ]
    return "M " + " L ".join(f"{px:.3f},{py:.3f}" for px, py in points) + " Z"


def _coral_scale(value, maximum):
    if value <= 0 or maximum <= 0:
        return WHITE
    low = np.array([245, 221, 220])
    high = np.array([242, 131, 120])
    ratio = 0.25 + 0.75 * np.sqrt(value / maximum)
    rgb = np.round(low + (high - low) * ratio).astype(int)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def make_hex_map(df, selected_state=None):
    grouped = df.groupby(["State", "Status"]).size().unstack(fill_value=0)
    maximum = int(grouped.get("Open", pd.Series(dtype=int)).max()) if not grouped.empty else 0
    shapes, xs, ys, labels, hover = [], [], [], [], []

    for state, (col, row) in HEX_COORDS.items():
        # Pointy-top grid spacing with a small, consistent white gutter.
        x = col * 0.80
        y = -row * 0.72
        opened = int(grouped.loc[state, "Open"]) if state in grouped.index and "Open" in grouped.columns else 0
        closed = int(grouped.loc[state, "Closed"]) if state in grouped.index and "Closed" in grouped.columns else 0
        selected = state == selected_state
        shapes.append(
            dict(
                type="path",
                path=_hex_path(x, y),
                fillcolor=_coral_scale(opened, maximum),
                line=dict(color="#8E8E8E" if selected else HEX_BORDER, width=2.5 if selected else 1.5),
                layer="below",
            )
        )
        xs.append(x)
        ys.append(y)
        labels.append(f"<b>{state}</b>")
        hover.append(f"<b>{state}</b><br>Open: {opened:,}<br>Closed: {closed:,}<br>Total: {opened + closed:,}")

    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=labels,
            textposition="middle center",
            textfont=dict(size=10, color="#8F8F8F", family="Montserrat, Segoe UI, Arial"),
            marker=dict(size=21, color="rgba(255,255,255,0)", line_width=0),
            customdata=list(HEX_COORDS.keys()),
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )
    fig.update_layout(
        **chart_layout(margin=dict(l=4, r=4, t=8, b=4)),
        shapes=shapes,
        xaxis=dict(visible=False, range=[-0.8, 9.8], fixedrange=True),
        yaxis=dict(visible=False, range=[-6.65, 0.65], scaleanchor="x", scaleratio=1, fixedrange=True),
    )
    return fig


def graph_config():
    return {"displayModeBar": False, "responsive": True}


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = os.environ.get("COMPLAINTS_CSV", BASE_DIR / "26k-consumer-complaints.csv")

try:
    DATA = prepare_data(load_data(CSV_PATH))
    LOAD_ERROR = None
except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
    DATA = pd.DataFrame()
    LOAD_ERROR = str(exc)

app = Dash(__name__, title="Complaints Dashboard", suppress_callback_exceptions=True)
server = app.server

if LOAD_ERROR:
    app.layout = html.Main(
        className="load-error",
        children=[
            html.H1("Complaints Dashboard"),
            html.H2("Unable to load complaint data"),
            html.P(LOAD_ERROR),
        ],
    )
else:
    DATE_MIN = DATA["Date received"].min().normalize()
    DATE_MAX = DATA["Date received"].max().normalize()
    DAY_MIN = DATE_MIN.toordinal()
    DAY_MAX = DATE_MAX.toordinal()
    source_options = [{"label": "All", "value": "All"}] + [
        {"label": value, "value": value} for value in sorted(DATA["Source Type"].unique())
    ]

    app.layout = html.Main(
        className="dashboard-shell",
        children=[
            dcc.Store(id="selected-state"),
            dcc.Store(id="selected-reason"),
            dcc.Store(id="selected-party"),
            html.Section(
                className="header-grid",
                children=[
                    html.Div(
                        className="hero",
                        children=[
                            html.H1("Complaints Dashboard"),
                            html.Div(
                                className="kpi-row",
                                children=[
                                    html.Span("Total Complaints:", className="kpi-lead"),
                                    html.Div([html.Span("Closed"), html.Strong(id="closed-kpi", className="closed-number")], className="kpi"),
                                    html.Div([html.Span("Open"), html.Strong(id="open-kpi", className="open-number")], className="kpi"),
                                    html.Div([html.Span("Total"), html.Strong(id="total-kpi", className="total-number")], className="kpi"),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="filters",
                        children=[
                            html.Label("Date Received"),
                            html.Div(
                                className="date-labels",
                                children=[html.Span(id="start-date-label"), html.Span(id="end-date-label")],
                            ),
                            dcc.RangeSlider(
                                id="date-range",
                                min=DAY_MIN,
                                max=DAY_MAX,
                                value=[DAY_MIN, DAY_MAX],
                                step=1,
                                marks=None,
                                allowCross=False,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            html.Div(
                                className="dropdown-row",
                                children=[
                                    html.Div([html.Label("Source Type"), dcc.Dropdown(id="source-filter", options=source_options, value="All", clearable=False)]),
                                    html.Div([html.Label("Show Open/Closed"), dcc.Dropdown(id="status-filter", options=["All", "Open", "Closed"], value="All", clearable=False)]),
                                ],
                            ),
                            html.Div(
                                className="filter-actions",
                                children=[
                                    html.Div(id="active-filters", className="active-filters"),
                                    html.Button("Clear Filters", id="clear-filters", n_clicks=0),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Section(
                className="chart-grid",
                children=[
                    html.Article(
                        className="chart-card",
                        children=[html.H2("Complaints by Month"), dcc.Graph(id="month-chart", config=graph_config(), className="dashboard-graph", style={"height": "270px"})],
                    ),
                    html.Article(
                        className="chart-card",
                        children=[
                            html.H2(["Open Complaints by State ", html.Em("(click to filter)")]),
                            dcc.Graph(id="state-map", config=graph_config(), className="dashboard-graph", style={"height": "270px"}),
                        ],
                    ),
                    html.Article(
                        className="chart-card",
                        children=[html.H2("Complaints by Reason"), dcc.Graph(id="reason-chart", config=graph_config(), className="dashboard-graph", style={"height": "270px"})],
                    ),
                    html.Article(
                        className="chart-card",
                        children=[
                            html.H2(["Complaints by Party ", html.Em("(click to filter)")]),
                            dcc.Graph(id="party-chart", config=graph_config(), className="dashboard-graph", style={"height": "270px"}),
                        ],
                    ),
                ],
            ),
        ],
    )


@app.callback(
    Output("selected-state", "data"),
    Output("selected-reason", "data"),
    Output("selected-party", "data"),
    Input("state-map", "clickData"),
    Input("reason-chart", "clickData"),
    Input("party-chart", "clickData"),
    Input("clear-filters", "n_clicks"),
    State("selected-state", "data"),
    State("selected-reason", "data"),
    State("selected-party", "data"),
    prevent_initial_call=True,
)
def update_click_filters(state_click, reason_click, party_click, _clear, selected_state, selected_reason, selected_party):
    trigger = ctx.triggered_id
    if trigger == "clear-filters":
        return None, None, None
    if trigger == "state-map" and state_click:
        value = state_click["points"][0]["customdata"]
        return (None if value == selected_state else value), selected_reason, selected_party
    if trigger == "reason-chart" and reason_click:
        value = reason_click["points"][0]["customdata"]
        return selected_state, (None if value == selected_reason else value), selected_party
    if trigger == "party-chart" and party_click:
        value = json.loads(party_click["points"][0]["customdata"])
        return selected_state, selected_reason, (None if selected_party and value["label"] == selected_party.get("label") else value)
    return selected_state, selected_reason, selected_party


@app.callback(
    Output("closed-kpi", "children"),
    Output("open-kpi", "children"),
    Output("total-kpi", "children"),
    Output("month-chart", "figure"),
    Output("state-map", "figure"),
    Output("reason-chart", "figure"),
    Output("party-chart", "figure"),
    Output("start-date-label", "children"),
    Output("end-date-label", "children"),
    Output("active-filters", "children"),
    Input("date-range", "value"),
    Input("source-filter", "value"),
    Input("status-filter", "value"),
    Input("selected-state", "data"),
    Input("selected-reason", "data"),
    Input("selected-party", "data"),
)
def render_dashboard(date_range, source_type, status, selected_state, selected_reason, selected_party):
    start = pd.Timestamp.fromordinal(date_range[0])
    end = pd.Timestamp.fromordinal(date_range[1])
    filtered = apply_filters(DATA, start, end, source_type, status, selected_state, selected_reason, selected_party)

    # Selector charts omit their own filter so users can change or toggle the selection.
    state_context = apply_filters(DATA, start, end, source_type, status, None, selected_reason, selected_party)
    reason_context = apply_filters(DATA, start, end, source_type, status, selected_state, None, selected_party)
    party_context = apply_filters(DATA, start, end, source_type, status, selected_state, selected_reason, None)

    closed, opened, total = make_kpi(filtered)
    chips = []
    if selected_state:
        chips.append(html.Span(f"State: {selected_state}"))
    if selected_reason:
        chips.append(html.Span(f"Reason: {selected_reason}"))
    if selected_party:
        chips.append(html.Span(f"Party: {selected_party['label']}"))

    return (
        f"{closed:,}",
        f"{opened:,}",
        f"{total:,}",
        make_month_chart(filtered, start, end),
        make_hex_map(state_context, selected_state),
        make_reason_chart(reason_context, selected_reason),
        make_party_chart(party_context, selected_party),
        start.strftime("%-m/%-d/%Y") if os.name != "nt" else f"{start.month}/{start.day}/{start.year}",
        end.strftime("%-m/%-d/%Y") if os.name != "nt" else f"{end.month}/{end.day}/{end.year}",
        chips,
    )


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", "8050")), debug=False)
