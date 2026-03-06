# visualization/dashboard.py - TELJES MŰKÖDŐ DASHBOARD (TE pontos struktúrával)

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class TradingDashboard:
    """
    Teljes körű Trading Dashboard - A TE backtest_results struktúrával
    """

    def __init__(self, config, backtest_results=None, backtest_data=None):
        self.config = config

        # TE pontos backtest_results struktúra
        self.backtest_results = backtest_results or self._generate_dummy_results_te()
        self.backtest_data = backtest_data

        # Dash app
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.app.title = "AI Trading Dashboard"

        # Layout és callback-ek
        self._setup_layout()
        self._setup_callbacks()

    def _generate_dummy_results_te(self):
        """Dummy adatok a TE pontos struktúrával"""
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=252, freq='B')

        # Portfolio values
        returns = np.random.normal(0.0008, 0.02, 252)
        portfolio_values = 100000 * np.cumprod(1 + returns)
        daily_returns = pd.Series(returns, index=dates)

        return {
            'total_return': (portfolio_values[-1] / 100000 - 1),
            'sharpe_ratio': 1.25,
            'max_drawdown': -0.125,
            'win_rate': 0.582,
            'final_portfolio_value': portfolio_values[-1],
            'total_trades': 87,
            'portfolio_values': portfolio_values,
            'daily_returns': daily_returns
        }

    def _setup_layout(self):
        """Teljes dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🤖 AI Trading Dashboard",
                        style={'textAlign': 'center', 'color': '#2ecc71'}),
                html.P("Portfolio teljesítmény elemzése",
                       style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': 18})
            ], style={'background': 'linear-gradient(90deg, #2ecc71, #27ae60)',
                      'color': 'white', 'padding': '20px', 'borderRadius': 10, 'marginBottom': 30}),

            # Row 1: Portfolio chart + Metrics
            html.Div([
                # Portfolio chart
                html.Div([
                    html.H3("📈 Portfolio Alakulása", style={'color': '#2ecc71', 'textAlign': 'center'}),
                    dcc.Graph(id='portfolio-chart')
                ], style={'flex': 2, 'padding': '20px'}),

                # Metrics cards - TE pontos kulcsokkal!
                html.Div([
                    html.H3("📊 Teljesítmény Mutatók", style={'color': '#3498db', 'textAlign': 'center'}),
                    html.Div(id='metrics-cards')
                ], style={'flex': 1, 'padding': '20px'})
            ], style={'display': 'flex', 'marginBottom': 30}),

            # Row 2: Daily Returns + Portfolio Values Table
            html.Div([
                html.Div([
                    html.H3("📉 Napi Hozam/Mínuszok", style={'color': '#e74c3c'}),
                    dcc.Graph(id='daily-returns-chart')
                ], style={'flex': 1, 'padding': '20px'}),

                html.Div([
                    html.H3("💰 Portfolio Érték Történet", style={'color': '#f39c12'}),
                    dash_table.DataTable(
                        id='portfolio-table',
                        columns=[
                            {"name": "Nap", "id": "step"},
                            {"name": "Portfolio ($)", "id": "value", "type": "numeric",
                             "format": {"specifier": ",.0f"}},
                            {"name": "Napi %", "id": "daily_pct", "type": "numeric", "format": {"specifier": ",.2f%"}}
                        ],
                        style_cell={'textAlign': 'right'},
                        style_data_conditional=[
                            {'if': {'filter_query': '{daily_pct} > 0'}, 'backgroundColor': '#d5f4e6'},
                            {'if': {'filter_query': '{daily_pct} < 0'}, 'backgroundColor': '#fadbd8'}
                        ],
                        page_size=20
                    )
                ], style={'flex': 1, 'padding': '20px'})
            ], style={'display': 'flex', 'marginBottom': 30}),

            # Auto-refresh
            dcc.Interval(id='interval-component', interval=30 * 1000, n_intervals=0)
        ], style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '20px'})

    def _setup_callbacks(self):
        """Callback-ek a TE pontos backtest_results-szal"""

        # 1. Portfolio chart - TE 'portfolio_values'
        @self.app.callback(
            Output('portfolio-chart', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_portfolio_chart(n):
            fig = go.Figure()

            # TE pontos kulcsok!
            portfolio_values = self.backtest_results['portfolio_values']
            dates = np.arange(len(portfolio_values))  # Step index

            # Portfolio görbe
            fig.add_trace(go.Scatter(
                x=dates, y=portfolio_values,
                mode='lines+markers',
                name='Portfolio Érték',
                line=dict(color='#2ecc71', width=4),
                marker=dict(size=3)
            ))

            # Kezdő tőke vonal
            fig.add_hline(
                y=self.config.INITIAL_BALANCE if hasattr(self.config, 'INITIAL_BALANCE') else 100000,
                line_dash="dash", line_color="gray",
                annotation_text="Kezdő tőke"
            )

            # Teljesítmény vonal
            fig.add_hline(
                y=self.backtest_results['final_portfolio_value'],
                line_dash="dot", line_color="#3498db",
                annotation_text=f"Végső érték: ${self.backtest_results['final_portfolio_value']:,.0f}"
            )

            fig.update_layout(
                title=f"Portfolio Alakulása (Total Return: {self.backtest_results['total_return'] * 100:.1f}%)",
                xaxis_title="Napok", yaxis_title="Portfolio Érték ($)",
                hovermode='x unified', template='plotly_white',
                height=500
            )

            return fig

        # 2. Metrics cards - TE pontos kulcsokkal!
        @self.app.callback(
            Output('metrics-cards', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_metrics(n):
            results = self.backtest_results  # TE struktúra

            cards = html.Div([
                # Total Return
                html.Div([
                    html.H2(f"{results['total_return'] * 100:.1f}%"),
                    html.P("Teljes Return", style={'color': '#7f8c8d'}),
                    html.Span("🟢", style={'fontSize': 30})
                ], style={
                    'background': 'linear-gradient(135deg, #2ecc71, #27ae60)',
                    'color': 'white', 'padding': '20px', 'borderRadius': 15,
                    'textAlign': 'center', 'margin': '10px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.15)'
                }),

                # Sharpe Ratio
                html.Div([
                    html.H2(f"{results['sharpe_ratio']:.2f}"),
                    html.P("Sharpe Ratio", style={'color': '#7f8c8d'}),
                    html.Span("📊", style={'fontSize': 30})
                ], style={
                    'background': 'linear-gradient(135deg, #3498db, #2980b9)',
                    'color': 'white', 'padding': '20px', 'borderRadius': 15,
                    'textAlign': 'center', 'margin': '10px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.15)'
                }),

                # Max Drawdown
                html.Div([
                    html.H2(f"{results['max_drawdown'] * 100:.1f}%"),
                    html.P("Max. Visszaesés", style={'color': '#7f8c8d'}),
                    html.Span("📉", style={'fontSize': 30})
                ], style={
                    'background': 'linear-gradient(135deg, #e74c3c, #c0392b)',
                    'color': 'white', 'padding': '20px', 'borderRadius': 15,
                    'textAlign': 'center', 'margin': '10px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.15)'
                }),

                # Total Trades
                html.Div([
                    html.H2(f"{results['total_trades']:,}"),
                    html.P("Tranzakciók", style={'color': '#7f8c8d'}),
                    html.Span("💼", style={'fontSize': 30})
                ], style={
                    'background': 'linear-gradient(135deg, #f39c12, #e67e22)',
                    'color': 'white', 'padding': '20px', 'borderRadius': 15,
                    'textAlign': 'center', 'margin': '10px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.15)'
                })
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'})

            return cards

        # 3. Daily Returns chart - TE 'daily_returns'
        @self.app.callback(
            Output('daily-returns-chart', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_daily_returns(n):
            fig = go.Figure()

            daily_returns = self.backtest_results['daily_returns']
            dates = np.arange(len(daily_returns))

            # Napi hozamok
            fig.add_trace(go.Bar(
                x=dates, y=daily_returns * 100,
                name='Napi Hozam (%)',
                marker_color=['green' if x >= 0 else 'red' for x in daily_returns * 100],
                opacity=0.7
            ))

            # Átlagvonal
            fig.add_hline(
                y=daily_returns.mean() * 100,
                line_dash="dash", line_color="orange",
                annotation_text=f"Átlag: {daily_returns.mean() * 100:.2f}%"
            )

            fig.update_layout(
                title="Napi Hozamok Eloszlása",
                xaxis_title="Napok", yaxis_title="Hozam (%)",
                template='plotly_white', height=400
            )

            return fig

        # 4. Portfolio table - TE 'portfolio_values'
        @self.app.callback(
            Output('portfolio-table', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_portfolio_table(n):
            portfolio_values = self.backtest_results['portfolio_values']
            daily_returns = self.backtest_results['daily_returns']

            table_data = []
            initial_value = portfolio_values[0]

            for i in range(min(100, len(portfolio_values))):  # Első 100 nap
                table_data.append({
                    'step': i + 1,
                    'value': portfolio_values[i],
                    'daily_pct': daily_returns[i] if i < len(daily_returns) else 0
                })

            return table_data

    def run(self, debug=False, port=8050):
        """Dashboard indítás"""
        print("🚀 AI Trading Dashboard indul: http://127.0.0.1:8050")
        print(f"📊 Backtest eredmények: {len(self.backtest_results['portfolio_values'])} nap")
        self.app.run(debug=debug, port=port, host='127.0.0.1')


# Indítás script
if __name__ == "__main__":
    from config.settings import Config

    config = Config()

    # Dummy backtest eredmények (cseréld ki a valódi backtest_agent hívásra)
    dummy_results = {
        'total_return': 0.187,
        'sharpe_ratio': 1.35,
        'max_drawdown': -0.092,
        'win_rate': 0.643,
        'final_portfolio_value': 118700,
        'total_trades': 124,
        'portfolio_values': np.cumsum(np.random.normal(400, 2000, 252)) + 100000,
        'daily_returns': pd.Series(np.random.normal(0.0008, 0.02, 252))
    }

    dashboard = TradingDashboard(config, backtest_results=dummy_results)
    dashboard.run(debug=True)
