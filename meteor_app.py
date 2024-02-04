import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FILENAME = "meteorites.csv"
MIN_YEAR = 1850
MAX_YEAR = 2013
SLIDER_STEPS = 10
TOP_COUNT = 50
TOP_STATS_COUNT = 20

def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Iterate through all the columns of a dataframe and modify the data type to reduce memory usage."""
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    logger.info(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB")
    return df


class MeteorDashApp:
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=["./assets/style.css"])
        self.load_data()
        self.create_visualizations()
        self.set_layout()

    def run_app(self):
        try:
            self.app.run_server(debug=True)
        except Exception as e:
            logger.exception(f"Error running the dashboard app: {e}")
            raise

    def load_data(self) -> None:
        try:
            raw_df = pd.read_csv(FILENAME, engine="c")
            raw_df = raw_df[(raw_df["year"] >= MIN_YEAR) & (raw_df["year"] <= MAX_YEAR)]
            raw_df = raw_df.dropna()
            # self.df = reduce_mem_usage(raw_df)
            self.df = raw_df
        except Exception as e:
            logger.exception(f"Error loading data: {e}")
            raise

    def reduce_mem_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            return reduce_mem_usage(df)
        except Exception as e:
            logger.exception(f"Error reducing memory usage: {e}")
            raise

    def create_visualizations(self) -> None:
        try:
            self.create_bar_chart()
            self.create_pie_chart()
            self.create_scatter_geo()
            self.create_box_plot()
            self.create_heatmap()
            self.create_lat_scatter()
            self.create_lon_scatter()
            self.create_top_classes_bar_chart()
            self.create_decade_bar_chart()
            self.create_stats_box_plot()
            self.create_top_mass_line_chart()
        except Exception as e:
            logger.exception(f"Error creating visualizations: {e}")
            raise

    def create_top_classes_bar_chart(self) -> None:
        try:
            top_classes = self.df["class"].value_counts().nlargest(TOP_COUNT)
            top_classes_bar_chart = px.bar(
                x=top_classes.index,
                y=top_classes.values,
                labels={"x": "Meteor Class", "y": "Count"},
                title=f"Top {TOP_COUNT} Meteor Classes",
            )
            self.top_classes_bar_chart = top_classes_bar_chart
        except Exception as e:
            logger.exception(f"Error creating top classes bar chart: {e}")
            raise
    
    def create_decade_bar_chart(self) -> None:
        try:
            self.df["decade"] = (self.df["year"] // 10) * 10
            meteor_counts_decade = self.df["decade"].value_counts().sort_index()
            decade_bar_chart = px.bar(
                x=meteor_counts_decade.index,
                y=meteor_counts_decade.values,
                labels={"x": "Decade", "y": "Meteor Count"},
                title="Meteor Counts Over the Decade",
            )
            decade_bar_chart.update_xaxes(type='category')
            self.decade_bar_chart = decade_bar_chart
        except Exception as e:
            logger.exception(f"Error creating decade line chart: {e}")
            raise

    def create_bar_chart(self) -> None:
        try:
            year_counts = self.df["year"].value_counts()
            self.bar_chart_data = px.bar(
                x=year_counts.index,
                y=year_counts.values,
                labels={"x": "Year", "y": "Meteor Count"},
                title="Meteor Falls Over the Years",
            )
        except Exception as e:
            logger.exception(f"Error creating bar chart: {e}")
            raise

    def create_pie_chart(self) -> None:
        try:
            fall_distribution = self.df["fall"].value_counts()
            self.pie_chart = go.Figure(
                data=[go.Pie(labels=fall_distribution.index, values=fall_distribution.values)]
            )
            self.pie_chart.update_layout(title_text="Fall Distribution")
        except Exception as e:
            logger.exception(f"Error creating pie chart: {e}")
            raise

    def create_scatter_geo(self) -> None:
        try:
            self.fig_world_map = px.scatter_geo(
                self.df,
                lat="lat",
                lon="long",
                size="mass",
                color="fall",
                title="Meteor Mass Worldwide",
                size_max=30,
                hover_name="name",
                hover_data={"lat": True, "long": True, "mass": True, "year": True},
            )
            self.fig_world_map.update_geos(
                projection_type="natural earth",
                showland=True,
                landcolor="rgb(243, 243, 243)",
                countrycolor="rgb(204, 204, 204)",
            )
        except Exception as e:
            logger.exception(f"Error creating scatter geo plot: {e}")
            raise

    # Inside the MeteorDashApp class
    def create_top_mass_line_chart(self) -> None:
        try:            
            top_masses = self.df.groupby(["year", "name"])["mass"].agg("sum").sort_values(ascending=False).head(TOP_COUNT)
            top_masses = top_masses.reset_index().sort_values(by="year")

            top_mass_line_chart = px.line(
                x=top_masses['year'],
                y=top_masses['mass'],
                labels={"x": "Year", "y": "Mass (in gm)"},
                title=f"Top {TOP_COUNT} Meteor Masses Over the Years",
                markers=True,
                line_shape="linear",
                hover_data={"name": top_masses['name']},
            )

            self.top_mass_line_chart = top_mass_line_chart
        except Exception as e:
            logger.exception(f"Error creating top mass line chart: {e}")
            raise

    def create_box_plot(self) -> None:
        try:
            self.box_plot = px.box(
                self.df, x="class", y="mass", title="Mass Distribution by Meteorite Class"
            )
        except Exception as e:
            logger.exception(f"Error creating box plot: {e}")
            raise

    def create_heatmap(self) -> None:
        try:
            correlation_matrix = self.df.drop(columns=["id"]).corr(numeric_only=True)
            self.heatmap_fig = go.Figure(
                data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                )
            )
            self.heatmap_fig.update_layout(title="Correlation Heatmap")
        except Exception as e:
            logger.exception(f"Error creating heatmap: {e}")
            raise

    def create_lat_scatter(self) -> None:
        try:
            # zero_lat_long_df = self.df[(self.df["lat"] != 0) & (self.df["long"] != 0)]
            lat_counts = self.df["lat"].value_counts().reset_index()
            lat_counts.columns = ["lat", "count"]
            self.lat_scatter = px.scatter(
                lat_counts,
                x="lat",
                y="count",
                title="Latitude Distribution",
                labels={"lat": "Latitude", "y": "Number of Records"},
            )
        except Exception as e:
            logger.exception(f"Error creating latitude scatter plot: {e}")
            raise

    def create_lon_scatter(self) -> None:
        try:
            # zero_lat_long_df = self.df[(self.df["lat"] != 0) & (self.df["long"] != 0)]
            lon_counts = self.df["long"].value_counts().reset_index()
            lon_counts.columns = ["long", "count"]
            self.lon_scatter = px.scatter(
                lon_counts,
                x="long",
                y="count",
                title="Longitude Distribution",
                labels={"long": "Longitude", "y": "Number of Records"},
            )
        except Exception as e:
            logger.exception(f"Error creating longitude scatter plot: {e}")
            raise

    def create_stats_box_plot(self) -> None:
        try:
            top_classes = self.df["class"].value_counts().nlargest(TOP_STATS_COUNT).index
            class_stats = self.df[self.df['class'].isin(top_classes)].groupby('class')['mass'].agg(['mean', 'median', 'std']).reset_index()
            
            # Create subplots
            fig = make_subplots(rows=2, cols=2, subplot_titles=['Mean', 'Median', 'Standard Deviation', 'Mass Distribution'])
            
            boxes = [(1,1),(1,2),(2,1),(2,2)]
            
            # Plot mean, median, and std in the first three subplots
            for i, stat in enumerate(['mean', 'median', 'std']):    
                trace = go.Bar(
                        x=class_stats['class'],
                        y=class_stats[stat],
                        name=stat.capitalize(),
                        text=class_stats[stat].round(2).astype(str),
                        textposition='outside'
                    )
                fig.add_trace(trace, row=boxes[i][0], col=boxes[i][1])
            
            # Plot mass distribution in the fourth subplot
            trace_mass_dist = px.box(self.df[self.df['class'].isin(top_classes)], x='class', y='mass')
            for trace in trace_mass_dist['data']:
                fig.add_trace(trace, row=2, col=2)
            
            # Update layout
            fig.update_layout(title="Meteorite Mass Statistics and Distribution by Class", showlegend=False)

            fig.update_layout(height=800)
            self.stats_box_plot = fig
        except Exception as e:
            logger.exception(f"Error creating box plot: {e}")
            raise

    def set_layout(self) -> None:
        try:
            self.app.layout = html.Div(
                [
                    html.Nav(
                        children=[
                            html.H1(
                                children="Meteor Data Insights",
                                style={
                                    "flex": "1",
                                    "textAlign": "center",
                                    "color": "white",
                                },
                            ),
                        ],
                        style={
                            "background-color": "#333",
                            "padding": "10px",
                            "display": "flex",
                            "justify-content": "center",
                            "align-items": "center",
                        },
                    ),
                    html.Div(
                        children=[
                            dcc.Graph(
                                id="scatter-geo",
                                figure=self.fig_world_map,
                                config=dict({"scrollZoom": False}),
                                style={"width": "100%", "height": "700px"},
                            ),
                            html.Div(
                                [  # Wrap dcc.Slider in html.Div
                                    self.create_chart_div(
                                        "year-slider",
                                        dcc.Slider(
                                            id="year-slider",
                                            min=self.df["year"].min(),
                                            max=self.df["year"].max(),
                                            value=self.df["year"].max(),
                                            marks={
                                                str(year): str(year)
                                                for year in range(
                                                    int(self.df["year"].min()),
                                                    int(self.df["year"].max()),
                                                    SLIDER_STEPS,
                                                )
                                            },
                                        ),
                                    ),
                                ],
                                style={
                                    "width": "50%",
                                    "margin-bottom": "10px",
                                    "padding-bottom": "15px",
                                    "margin-top": "-30px",
                                    "display": "block",
                                    "margin-left": "auto",
                                    "margin-right": "auto",
                                },
                            ),
                        ],
                        style={"background": "#FFF", "margin": "20px"},
                    ),
                    self.create_chart_div("meteor-bar-chart", self.bar_chart_data),
                    self.create_chart_div("meteor-pie-chart", self.pie_chart),
                    self.create_chart_div("mass-distribution-plot", self.box_plot),
                    self.create_chart_div("correlation-heatmap", self.heatmap_fig),
                    self.create_chart_div("lat-scatter", self.lat_scatter),
                    self.create_chart_div("lon-scatter", self.lon_scatter),
                    self.create_chart_div("top-classes-bar-chart", self.top_classes_bar_chart),
                    self.create_chart_div("top-mass-line-chart", self.top_mass_line_chart),
                    self.create_chart_div("decade-line-chart", self.decade_bar_chart),
                    self.create_chart_div("mass-stats-bar-chart", self.stats_box_plot),
                ],
                style={
                    "width": "100%",
                    "background-color": "#f4f4f4",
                    "margin": "0",
                    "padding": "0",
                },
            )
        except Exception as e:
            logger.exception(f"Error setting layout: {e}")
            raise

    def create_chart_div(self, chart_id: str, chart_object) -> html.Div:
        try:
            if isinstance(chart_object, dcc.Slider):
                return html.Div(
                    children=[
                        chart_object
                    ],
                    style={"background": "#FFF", "margin": "20px"},
                )
            else:
                return html.Div(
                    children=[
                        dcc.Graph(
                            id=chart_id,
                            figure=chart_object,
                            style={"margin": "20px"},
                        )
                    ],
                    style={"background": "#FFF", "margin": "20px"},
                )
        except Exception as e:
            logger.exception(f"Error creating chart div: {e}")
            raise


meteor_dash_app = MeteorDashApp()

@meteor_dash_app.app.callback(Output("scatter-geo", "figure"), [Input("year-slider", "value")])
def update_map(selected_year):
    try:
        filtered_data = meteor_dash_app.df[meteor_dash_app.df["year"] <= selected_year]
        fig_world_map = px.scatter_geo(
            filtered_data,
            lat="lat",
            lon="long",
            size="mass",
            color="fall",
            title=f"Meteor Mass Worldwide (Up to {selected_year})",
            size_max=30,
            hover_name="name",
            hover_data={"lat": True, "long": True, "mass": True, "year": True},
        )
        fig_world_map.update_geos(
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(243, 243, 243)",
            countrycolor="rgb(204, 204, 204)",
        )
        fig_world_map.update_layout(title=f"Meteor Mass Worldwide (Up to {selected_year})")
        return fig_world_map
    except Exception as e:
        logger.exception(f"Error updating map: {e}")
        raise

if __name__ == "__main__":
    meteor_dash_app.run_app()
