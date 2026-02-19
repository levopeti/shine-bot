# visualization/dashboard.py

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from typing import Dict


class TradingDashboard:
    """
    Real-time trading dashboard using Dash and Plotly
    """

    def __init__(self, config):
        self.config = config
        self.app = dash.Dash(__name__)
        self.setup_layout()

    def setup_layout(self):
        """
        Setup dashboard layout
        """
        self.app.layout = html.Div([
            html.H1("AI Trading Framework Dashboard", style={'textAlign': 'center'}),

            # Portfolio value chart
            html.Div([
                html.H3("Portfolio Value Over Time"),
                dcc.Graph(id='portfolio-chart')
            ]),

            # Performance metrics
            html.Div([
                html.H3("Performance Metrics"),
                html.Div(id='metrics-display')
            ]),

            # Asset allocation
            html.Div([
                html.H3("Current Asset Allocation"),
                dcc.Graph(id='allocation-chart')
            ]),

            # Price charts
            html.Div([
                html.H3("Asset Price Comparison"),
                dcc.Dropdown(
                    id='asset-selector',
                    multi=True,
                    placeholder="Select assets to compare"
                ),
                dcc.Graph(id='price-chart')
            ]),

            # Update interval
            dcc.Interval(
                id='interval-component',
                interval=self.config.UPDATE_INTERVAL,
                n_intervals=0
            )
        ])

    def create_portfolio_chart(self, portfolio_values: np.ndarray):
        """
        Create portfolio value line chart
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#2ecc71', width=2)
        ))

        fig.add_hline(
            y=self.config.INITIAL_BALANCE,
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Balance"
        )

        fig.update_layout(
            xaxis_title="Time Steps",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            template='plotly_dark'
        )

        return fig

    def create_metrics_display(self, metrics: Dict) -> html.Div:
        """
        Create metrics display
        """
        return html.Div([
            html.Div([
                html.P(f"Total Return: {metrics['total_return']*100:.2f}%"),
                html.P(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}"),
                html.P(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%"),
                html.P(f"Win Rate: {metrics['win_rate']*100:.2f}%"),
                html.P(f"Total Trades: {metrics['total_trades']}")
            ], style={'fontSize': 18, 'padding': '20px'})
        ])

    def create_allocation_chart(self, positions: Dict):
        """
        Create pie chart for asset allocation
        """
        labels = list(positions.keys())
        values = list(positions.values())

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3
        )])

        fig.update_layout(template='plotly_dark')

        return fig

    def create_price_comparison_chart(self, data: pd.DataFrame, selected_assets: list):
        """
        Create price comparison chart for multiple assets
        """
        fig = go.Figure()

        for asset in selected_assets:
            # Find close price column for this asset
            close_cols = [col for col in data.columns if asset in col and 'Close' in col]
            if close_cols:
                prices = data[close_cols[0]]

                # Normalize to percentage change from start
                normalized_prices = (prices / prices.iloc[0] - 1) * 100

                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=normalized_prices,
                    mode='lines',
                    name=asset
                ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price Change (%)",
            hovermode='x unified',
            template='plotly_dark'
        )

        return fig

    def run(self, debug=False):
        """
        Run the dashboard server
        """
        self.app.run(
            debug=debug,
            port=self.config.DASHBOARD_PORT
        )
