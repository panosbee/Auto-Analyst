"""
Enhanced Statistical Agent for Auto-Analyst
--------------------------------------------

This module defines the EnhancedStatisticalAgent, specializing in advanced
statistical analyses such as time series analysis, complex hypothesis testing,
correlation networks, and anomaly detection.
"""

import dspy
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.eval_measures import rmse, aic
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import streamlit as st
import matplotlib.pyplot as plt # For ACF/PACF plots specifically if Plotly versions are tricky
import networkx as nx # For correlation network
from sklearn.preprocessing import StandardScaler # For DBSCAN in anomaly detection
from sklearn.cluster import DBSCAN # For anomaly detection
from sklearn.decomposition import PCA # For visualizing DBSCAN results
from sklearn.metrics import mean_squared_error # For ARIMA evaluation

class EnhancedStatisticalAgent(dspy.Signature):
    """
    Advanced statistical analysis agent that performs sophisticated statistical
    analysis including time series, hypothesis testing, correlation networks,
    distribution analysis, and anomaly detection.

    Capabilities:
    - Time series analysis (ARIMA, seasonal decomposition, trend analysis, stationarity tests, forecasting).
    - Comprehensive correlation analysis (Pearson, Spearman, Kendall, VIF, network graphs).
    - Statistical hypothesis testing (normality, t-tests, ANOVA, Chi-square, with visualizations).
    - Anomaly detection using statistical methods (Z-score, IQR, multivariate with DBSCAN).
    - Advanced distribution analysis and fitting (covered within hypothesis testing and anomaly detection).
    """
    dataset_description = dspy.InputField(desc="Schema of the DataFrame `df` (columns, dtypes, basic stats).")
    user_goal = dspy.InputField(desc="The user-defined goal for the statistical analysis (e.g., 'analyze sales trend', 'find anomalies in sensor data', 'test if new feature improved conversion').")
    hint = dspy.InputField(desc="Optional context from previous analysis steps or user clarifications.", default="")
    code = dspy.OutputField(desc="Python code that performs the requested advanced statistical analysis. The code should use `st.write` for text and `st.plotly_chart` or `st.pyplot` for plots.")
    commentary = dspy.OutputField(desc="Detailed explanation of the statistical methods employed, key assumptions, and interpretation of the results and insights derived.")
    kpis = dspy.OutputField(desc="Key statistical metrics, test statistics, p-values, and other relevant indicators identified during the analysis (e.g., 'ADF p-value: 0.02', 'Correlation (A,B): 0.85', 'Anomalies found: 15').", default="")

    # Note: The actual implementation of how dspy.Module uses these static methods
    # would be in the `forward` method of a class that *uses* this signature.
    # For example, a `StatisticalAnalysisModule(dspy.Module)` would call these.
    # Here, they are provided as helper code generators.

    @staticmethod
    def time_series_analysis(df_placeholder_name="df", target_col_placeholder="target_column_name", date_col_placeholder="date_column_name", freq_placeholder="None", seasonal_periods_placeholder="None"):
        """
        Generates Python code for comprehensive time series analysis.
        Assumes `df` is the DataFrame name, `target_col` is the series to analyze,
        and `date_col` is the time index.
        """
        # Placeholders are used so the agent can fill them based on actual column names from dataset_description
        code = f"""
# Time Series Analysis for column: '{target_col_placeholder}'
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import streamlit as st

# Ensure the DataFrame '{df_placeholder_name}' is available and the columns exist.
# df = st.session_state.df # Or however df is made available to the exec scope

target_col = '{target_col_placeholder}'
date_col = '{date_col_placeholder}' # If no specific date column, agent might need to infer or use index

# Prepare data
if date_col in {df_placeholder_name}.columns:
    # Ensure date column is datetime
    try:
        {df_placeholder_name}[date_col] = pd.to_datetime({df_placeholder_name}[date_col])
    except Exception as e:
        st.warning(f"Could not convert '{{date_col}}' to datetime: {{e}}. Attempting to use index if it's DatetimeIndex.")
        if not isinstance({df_placeholder_name}.index, pd.DatetimeIndex):
            st.error(f"Date column '{{date_col}}' could not be converted, and index is not DatetimeIndex. Time series analysis cannot proceed.")
            st.stop() # Or return to prevent further execution
    
    # Set date column as index if it's not already
    if {df_placeholder_name}.index.name != date_col:
        try:
            {df_placeholder_name} = {df_placeholder_name}.set_index(date_col)
        except Exception as e:
            st.error(f"Could not set '{{date_col}}' as index: {{e}}")
            st.stop()

elif isinstance({df_placeholder_name}.index, pd.DatetimeIndex):
    st.write(f"Using the DataFrame's DatetimeIndex for time series analysis.")
else:
    st.error(f"No suitable date column ('{{date_col}}') found or DataFrame index is not a DatetimeIndex. Time series analysis requires a time dimension.")
    st.stop()

# Sort index just in case
{df_placeholder_name} = {df_placeholder_name}.sort_index()

# Select the target series
if target_col not in {df_placeholder_name}.columns:
    st.error(f"Target column '{{target_col}}' not found in the DataFrame.")
    st.stop()

ts_data = {df_placeholder_name}[target_col].dropna() # Drop NA for analysis

if ts_data.empty:
    st.warning(f"Time series for '{{target_col}}' is empty after dropping NAs. Cannot perform analysis.")
    st.stop()
if len(ts_data) < 10: # Arbitrary small number
    st.warning(f"Time series for '{{target_col}}' has very few data points ({{len(ts_data)}}). Results might be unreliable.")


st.write(f"### Time Series Analysis: {{target_col}}")

# Plot the time series
fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(x=ts_data.index, y=ts_data, mode='lines', name=target_col))
fig_ts.update_layout(title=f'Time Series Plot of {{target_col}}', xaxis_title='Time', yaxis_title=target_col)
st.plotly_chart(fig_ts, use_container_width=True)

# Stationarity Test (ADF)
st.write("#### Stationarity Test (Augmented Dickey-Fuller)")
adf_result = adfuller(ts_data)
st.write(f'ADF Statistic: {{adf_result[0]:.4f}}')
st.write(f'p-value: {{adf_result[1]:.4f}}')
st.write(f'Critical Values: {{adf_result[4]}}')
is_stationary = adf_result[1] < 0.05
st.write(f"**Conclusion:** The series is likely **{{'stationary' if is_stationary else 'non-stationary'}}** at a 5% significance level.")

# Seasonal Decomposition
st.write("#### Seasonal Decomposition")
# Determine seasonal period: try common ones or use provided, default to 12 if enough data
seasonal_period_val = {seasonal_periods_placeholder}
if seasonal_period_val is None or not isinstance(seasonal_period_val, int) or seasonal_period_val <= 1:
    if len(ts_data) >= 24: seasonal_period_val = 12 # Monthly default
    elif len(ts_data) >= 8: seasonal_period_val = 4  # Quarterly default
    elif len(ts_data) >= 14: seasonal_period_val = 7 # Daily default for weekly seasonality
    else: seasonal_period_val = max(2, int(len(ts_data) / 2)) if len(ts_data) >= 4 else None # Fallback

if seasonal_period_val and len(ts_data) >= 2 * seasonal_period_val:
    try:
        decomposition = seasonal_decompose(ts_data, model='additive', period=seasonal_period_val)
        fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                   subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'))
        fig_decomp.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observed'), row=1, col=1)
        fig_decomp.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
        fig_decomp.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
        fig_decomp.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)
        fig_decomp.update_layout(height=700, title_text=f"Seasonal Decomposition (Period: {{seasonal_period_val}})")
        st.plotly_chart(fig_decomp, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not perform seasonal decomposition with period {{seasonal_period_val}}: {{e}}")
else:
    st.write("Not enough data or no suitable period for seasonal decomposition.")

# ACF and PACF plots
st.write("#### Autocorrelation (ACF) and Partial Autocorrelation (PACF) Plots")
lags_to_plot = min(40, len(ts_data) // 2 -1)
if lags_to_plot > 0:
    fig_acf_pacf, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_acf(ts_data, ax=ax1, lags=lags_to_plot)
    ax1.set_title('Autocorrelation Function (ACF)')
    plot_pacf(ts_data, ax=ax2, lags=lags_to_plot, method='ywm') # 'ywm' is often preferred
    ax2.set_title('Partial Autocorrelation Function (PACF)')
    plt.tight_layout()
    st.pyplot(fig_acf_pacf)
else:
    st.write("Not enough data points to plot ACF/PACF.")

# ARIMA Modeling (Example: Auto ARIMA or simple parameters)
st.write("#### ARIMA Modeling")
if len(ts_data) < 20:
    st.warning("Not enough data for reliable ARIMA modeling (need at least ~20 points).")
else:
    # A simple approach: determine d based on stationarity, then try common p, q
    d_order = 0
    if not is_stationary:
        # Check if first differencing makes it stationary
        ts_diff1 = ts_data.diff().dropna()
        if not ts_diff1.empty:
            adf_result_diff1 = adfuller(ts_diff1)
            if adf_result_diff1[1] < 0.05:
                d_order = 1
                st.write("Series appears to be I(1) - integrated of order 1.")
            else: # Try second differencing if needed (less common for typical business data)
                ts_diff2 = ts_diff1.diff().dropna()
                if not ts_diff2.empty:
                    adf_result_diff2 = adfuller(ts_diff2)
                    if adf_result_diff2[1] < 0.05:
                        d_order = 2
                        st.write("Series appears to be I(2) - integrated of order 2.")
                    else:
                        st.warning("Series may require more complex differencing or transformation for ARIMA.")
                else:
                    st.warning("Series became empty after second differencing.")
        else:
            st.warning("Series became empty after first differencing.")


    # Simple p, q based on ACF/PACF (visual inspection or common low values)
    # For automation, this is tricky. A common starting point is (1,d,1) or (2,d,2)
    # Or use auto_arima if available and allowed. Here, let's try a simple (1,d,1)
    p_order, q_order = 1, 1
    st.write(f"Attempting ARIMA(p={{p_order}}, d={{d_order}}, q={{q_order}}) model.")

    try:
        model = SARIMAX(ts_data, order=(p_order, d_order, q_order), seasonal_order=(0,0,0,0)) # Non-seasonal ARIMA
        results = model.fit(disp=False)
        st.write(results.summary())

        # Plot diagnostics
        fig_diag = results.plot_diagnostics(figsize=(15, 12))
        st.pyplot(fig_diag)

        # Forecast
        forecast_steps = min(len(ts_data) // 5, 24) # Forecast ~20% or max 24 steps
        if forecast_steps > 0:
            forecast_obj = results.get_forecast(steps=forecast_steps)
            forecast_mean = forecast_obj.predicted_mean
            forecast_ci = forecast_obj.conf_int()

            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=ts_data.index, y=ts_data, mode='lines', name='Historical Data'))
            fig_forecast.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean, mode='lines', name='Forecast', line=dict(dash='dash')))
            fig_forecast.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 0], mode='lines', line=dict(width=0), showlegend=False))
            fig_forecast.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 1], mode='lines', line=dict(width=0),
                                            fill='tonexty', fillcolor='rgba(0,176,246,0.2)', name='95% Confidence Interval'))
            fig_forecast.update_layout(title=f'ARIMA Forecast for {{target_col}}', xaxis_title='Time', yaxis_title=target_col)
            st.plotly_chart(fig_forecast, use_container_width=True)

            # Evaluation metrics (in-sample for simplicity, or on a hold-out set if data was split)
            # Here, calculate RMSE on fitted values for the observed period
            fitted_values = results.fittedvalues
            # Align indices if differencing was done, fittedvalues might be shorter
            common_index = ts_data.index.intersection(fitted_values.index)
            if not common_index.empty:
                rmse_val = np.sqrt(mean_squared_error(ts_data[common_index], fitted_values[common_index]))
                st.write(f"Model Fit RMSE: {{rmse_val:.4f}}")
            st.write(f"AIC: {{results.aic:.4f}}, BIC: {{results.bic:.4f}}")
        else:
            st.write("Not enough data to generate a forecast.")

    except Exception as e:
        st.error(f"Error fitting ARIMA model: {{e}}")
        st.write("This could be due to non-stationarity, inappropriate (p,d,q) orders, or other data issues.")
"""
        return code

    @staticmethod
    def correlation_analysis(df_placeholder_name="df"):
        """Generates Python code for comprehensive correlation analysis."""
        code = f"""
# Comprehensive Correlation Analysis
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm # For VIF constant
import streamlit as st

# df = st.session_state.df # Or however df is made available

numeric_cols = {df_placeholder_name}.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) < 2:
    st.warning("Correlation analysis requires at least two numeric columns.")
    st.stop()

st.write("### Correlation Analysis")
corr_df = {df_placeholder_name}[numeric_cols]

# Pearson Correlation (Linear)
st.write("#### Pearson Correlation (Linear Relationships)")
pearson_corr_matrix = corr_df.corr(method='pearson')
fig_pearson = px.imshow(pearson_corr_matrix, text_auto=".2f", aspect="auto",
                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                        title='Pearson Correlation Matrix')
st.plotly_chart(fig_pearson, use_container_width=True)

# Spearman Correlation (Monotonic)
st.write("#### Spearman Rank Correlation (Monotonic Relationships)")
spearman_corr_matrix = corr_df.corr(method='spearman')
fig_spearman = px.imshow(spearman_corr_matrix, text_auto=".2f", aspect="auto",
                         color_continuous_scale='Viridis', zmin=-1, zmax=1,
                         title='Spearman Rank Correlation Matrix')
st.plotly_chart(fig_spearman, use_container_width=True)

# Kendall Tau Correlation (Ordinal Association)
st.write("#### Kendall Tau Correlation (Ordinal Association)")
kendall_corr_matrix = corr_df.corr(method='kendall')
fig_kendall = px.imshow(kendall_corr_matrix, text_auto=".2f", aspect="auto",
                        color_continuous_scale='Cividis', zmin=-1, zmax=1,
                        title='Kendall Tau Correlation Matrix')
st.plotly_chart(fig_kendall, use_container_width=True)


# Highly Correlated Pairs
st.write("#### Highly Correlated Feature Pairs")
threshold = 0.7 # User might want to configure this
highly_correlated_pairs = []
for i in range(len(pearson_corr_matrix.columns)):
    for j in range(i):
        if abs(pearson_corr_matrix.iloc[i, j]) > threshold:
            pair_info = (pearson_corr_matrix.columns[i], pearson_corr_matrix.columns[j], pearson_corr_matrix.iloc[i, j])
            highly_correlated_pairs.append(pair_info)

if highly_correlated_pairs:
    hc_df = pd.DataFrame(highly_correlated_pairs, columns=['Feature 1', 'Feature 2', 'Pearson Correlation'])
    hc_df = hc_df.sort_values(by='Pearson Correlation', key=abs, ascending=False)
    st.write(f"Found {{len(hc_df)}} pairs with |Pearson Correlation| > {{threshold}}:")
    st.dataframe(hc_df)

    # Correlation Network Graph (for highly correlated pairs)
    st.write("##### Correlation Network Graph (Pearson)")
    G = nx.Graph()
    for _, row in hc_df.iterrows():
        G.add_edge(row['Feature 1'], row['Feature 2'], weight=abs(row['Pearson Correlation']), title=f"{{row['Pearson Correlation']:.2f}}")

    if G.nodes:
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42) # k adjusts spacing
        
        edge_x, edge_y = [], []
        edge_weights = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

        node_x, node_y, node_text, node_size = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_size.append(G.degree(node) * 5 + 10) # Size by degree

        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center",
                                marker=dict(showscale=False, size=node_size, color=node_size, colorscale='YlGnBu'),
                                hoverinfo='text', customdata=[f"Degree: {{G.degree(node)}}" for node in G.nodes()])
        
        fig_network = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(title='Feature Correlation Network (|r| > {{threshold}})', showlegend=False, hovermode='closest',
                                         margin=dict(b=20,l=5,r=5,t=40),
                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        st.plotly_chart(fig_network, use_container_width=True)
    else:
        st.write("No pairs found for network graph with current threshold.")
else:
    st.write(f"No pairs found with |Pearson Correlation| > {{threshold}}.")


# Scatter Plot Matrix for a subset of variables
st.write("#### Scatter Plot Matrix")
cols_for_scatter_matrix = numeric_cols[:min(6, len(numeric_cols))] # Limit to 6 for readability
if len(cols_for_scatter_matrix) >= 2:
    fig_scatter_matrix = px.scatter_matrix(corr_df, dimensions=cols_for_scatter_matrix,
                                           title=f"Scatter Matrix for: {{', '.join(cols_for_scatter_matrix)}}")
    fig_scatter_matrix.update_traces(diagonal_visible=False) # Hide histograms on diagonal for cleaner look
    st.plotly_chart(fig_scatter_matrix, use_container_width=True)
else:
    st.write("Not enough columns for a scatter matrix.")

# Variance Inflation Factor (VIF) for Multicollinearity
st.write("#### Variance Inflation Factor (VIF) for Multicollinearity")
if len(numeric_cols) >= 2:
    X_vif = corr_df.dropna() # VIF cannot handle NaNs
    if X_vif.empty or X_vif.shape[0] < X_vif.shape[1]: # Need more rows than columns
        st.warning("Not enough data after dropping NaNs, or too few rows for VIF calculation.")
    else:
        # Add constant for intercept, VIF calculation requires it
        X_vif_const = sm.add_constant(X_vif, prepend=False) # Add constant at the end
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_vif_const.columns[:-1] # Exclude the constant column itself
        vif_data["VIF"] = [variance_inflation_factor(X_vif_const.values, i) for i in range(X_vif_const.shape[1] - 1)]
        vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)
        
        st.write("VIF values (typically VIF > 5 or 10 indicates problematic multicollinearity):")
        st.dataframe(vif_data)

        fig_vif = px.bar(vif_data, x='Feature', y='VIF', title='Variance Inflation Factor (VIF)',
                         color='VIF', color_continuous_scale='OrRd')
        fig_vif.add_hline(y=5, line_dash="dash", line_color="blue", annotation_text="VIF=5 (Moderate)")
        fig_vif.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="VIF=10 (High)")
        st.plotly_chart(fig_vif, use_container_width=True)
else:
    st.write("Not enough numeric columns for VIF calculation.")
"""
        return code

    @staticmethod
    def hypothesis_testing(df_placeholder_name="df"):
        """Generates Python code for various hypothesis tests."""
        code = f"""
# Statistical Hypothesis Testing
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm # For QQ plot and ANOVA related
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# df = st.session_state.df # Or however df is made available

st.write("### Statistical Hypothesis Testing")
alpha = 0.05 # Significance level

numeric_cols = {df_placeholder_name}.select_dtypes(include=np.number).columns.tolist()
categorical_cols = {df_placeholder_name}.select_dtypes(include=['object', 'category']).columns.tolist()

# 1. Normality Tests for Numeric Columns
st.write("#### 1. Normality Tests (Shapiro-Wilk)")
st.write(f"Testing if numeric columns follow a normal distribution (significance level alpha = {{alpha}}).")
normality_results = []
for col in numeric_cols:
    data_to_test = {df_placeholder_name}[col].dropna()
    if len(data_to_test) >= 3 and len(data_to_test) <= 5000: # Shapiro-Wilk constraints
        stat, p_value = stats.shapiro(data_to_test)
        normality_results.append({
            'Column': col, 'Statistic': stat, 'p-value': p_value,
            'Is Normal (alpha={{alpha}})': p_value > alpha
        })
    elif len(data_to_test) > 5000:
         # For larger samples, D'Agostino's K^2 test is often recommended
        try:
            stat, p_value = stats.normaltest(data_to_test)
            normality_results.append({
                'Column': col, 'Statistic': stat, 'p-value': p_value,
                'Is Normal (alpha={{alpha}})': p_value > alpha, 'Test': "D'Agostino's K^2"
            })
        except Exception as e:
            normality_results.append({'Column': col, 'Statistic': np.nan, 'p-value': np.nan, 'Is Normal (alpha={{alpha}})': 'Error', 'Test': 'Error in K^2 test'})


if normality_results:
    normality_df = pd.DataFrame(normality_results)
    st.dataframe(normality_df)

    # Q-Q Plots for a few columns
    st.write("##### Q-Q Plots for Normality Visual Check")
    cols_to_plot_qq = normality_df['Column'].unique()[:min(4, len(normality_df))]
    if len(cols_to_plot_qq) > 0:
        num_qq_rows = (len(cols_to_plot_qq) + 1) // 2
        fig_qq = make_subplots(rows=num_qq_rows, cols=2, subplot_titles=[f"Q-Q Plot: {{c}}" for c in cols_to_plot_qq])
        for i, col_name in enumerate(cols_to_plot_qq):
            row, col_idx = i // 2 + 1, i % 2 + 1
            qq_data = {df_placeholder_name}[col_name].dropna()
            if len(qq_data) > 0:
                # Using statsmodels for qqplot data generation is robust
                qqplot_data = sm.ProbPlot(qq_data).qqplot(line='s', ax=None, fit=True) # Generates its own plot if ax not given
                # For Plotly, we need to extract points
                theoretical_q = qqplot_data.gca().get_lines()[0].get_xdata()
                sample_q = qqplot_data.gca().get_lines()[0].get_ydata()
                fit_line_q = qqplot_data.gca().get_lines()[1].get_ydata() # Fitted line
                plt.close(qqplot_data) # Close matplotlib plot

                fig_qq.add_trace(go.Scatter(x=theoretical_q, y=sample_q, mode='markers', name='Data Quantiles'), row=row, col=col_idx)
                fig_qq.add_trace(go.Scatter(x=theoretical_q, y=fit_line_q, mode='lines', name='Fit Line', line=dict(color='red')), row=row, col=col_idx)
                fig_qq.update_xaxes(title_text="Theoretical Quantiles", row=row, col=col_idx)
                fig_qq.update_yaxes(title_text="Sample Quantiles", row=row, col=col_idx)
        fig_qq.update_layout(height=300 * num_qq_rows, title_text="Q-Q Plots")
        st.plotly_chart(fig_qq, use_container_width=True)
else:
    st.write("No numeric columns suitable for normality testing, or tests failed.")


# 2. Two-Sample T-tests / ANOVA for Numeric vs. Categorical (with 2 or more groups)
st.write("#### 2. Group Difference Tests (Numeric vs. Categorical)")
if numeric_cols and categorical_cols:
    # Select one numeric and one categorical column for demonstration or let user choose
    # For automation, pick a common scenario: first numeric vs. first categorical with 2-5 groups
    
    num_col_test = numeric_cols[0] # Example
    cat_col_test = None
    for c in categorical_cols:
        if {df_placeholder_name}[c].nunique() >= 2 and {df_placeholder_name}[c].nunique() <= 5: # Good for ANOVA/t-test demo
            cat_col_test = c
            break
    
    if cat_col_test:
        st.write(f"Testing differences in '{{num_col_test}}' across groups in '{{cat_col_test}}'.")
        
        # Box plot for visual inspection
        fig_box_group = px.box({df_placeholder_name}, x=cat_col_test, y=num_col_test, points="all",
                               title=f"Distribution of {{num_col_test}} by {{cat_col_test}}")
        st.plotly_chart(fig_box_group, use_container_width=True)

        groups_data = [{df_placeholder_name}[{df_placeholder_name}[cat_col_test] == group][num_col_test].dropna() for group in {df_placeholder_name}[cat_col_test].unique()]
        groups_data = [g for g in groups_data if len(g) >= 2] # Ensure groups have enough data

        if len(groups_data) == 2:
            st.write(f"Performing Two-Sample T-test (Welch's t-test for unequal variances assumed):")
            t_stat, p_val_t = stats.ttest_ind(groups_data[0], groups_data[1], equal_var=False)
            st.write(f"T-statistic: {{t_stat:.4f}}, p-value: {{p_val_t:.4f}}")
            if p_val_t < alpha:
                st.write(f"**Conclusion:** Significant difference found between the two groups (p < {{alpha}}).")
            else:
                st.write(f"**Conclusion:** No significant difference found between the two groups (p >= {{alpha}}).")
        elif len(groups_data) > 2:
            st.write(f"Performing ANOVA (Analysis of Variance):")
            f_stat, p_val_anova = stats.f_oneway(*groups_data)
            st.write(f"F-statistic: {{f_stat:.4f}}, p-value: {{p_val_anova:.4f}}")
            if p_val_anova < alpha:
                st.write(f"**Conclusion:** Significant difference found among at least two groups (p < {{alpha}}).")
                st.write("Performing Tukey's HSD post-hoc test for pairwise comparisons:")
                # Ensure data is in a suitable format for pairwise_tukeyhsd
                tukey_data = {df_placeholder_name}[[num_col_test, cat_col_test]].dropna()
                if len(tukey_data) > 0:
                    tukey_results = pairwise_tukeyhsd(tukey_data[num_col_test], tukey_data[cat_col_test], alpha=alpha)
                    st.text(str(tukey_results))
                    # Plot Tukey's results (optional, can be complex to render nicely in Streamlit text)
                    # fig_tukey = tukey_results.plot_simultaneous()
                    # st.pyplot(fig_tukey) # Matplotlib plot
                    # plt.close() # Close the matplotlib plot
                else:
                    st.warning("Not enough data for Tukey's HSD after dropping NAs.")
            else:
                st.write(f"**Conclusion:** No significant difference found among the groups (p >= {{alpha}}).")
        else:
            st.write(f"Not enough groups ({{len(groups_data)}}) with sufficient data in '{{cat_col_test}}' for {{num_col_test}} to perform t-test or ANOVA.")
    else:
        st.write("Could not find a suitable categorical column (2-5 unique groups) for demonstration.")
else:
    st.write("Need at least one numeric and one categorical column for group difference tests.")


# 3. Chi-Square Test for Independence (Categorical vs. Categorical)
st.write("#### 3. Chi-Square Test of Independence (Categorical vs. Categorical)")
if len(categorical_cols) >= 2:
    # Select two categorical columns for demonstration
    cat1_test = categorical_cols[0]
    cat2_test = categorical_cols[1] if len(categorical_cols) > 1 else None

    if cat2_test:
        st.write(f"Testing independence between '{{cat1_test}}' and '{{cat2_test}}'.")
        contingency_table = pd.crosstab({df_placeholder_name}[cat1_test], {df_placeholder_name}[cat2_test])
        
        st.write("Contingency Table:")
        st.dataframe(contingency_table)

        # Heatmap of contingency table
        fig_chi_heat = px.imshow(contingency_table, text_auto=True, aspect="auto",
                                 title=f"Heatmap of Contingency: {{cat1_test}} vs {{cat2_test}}")
        st.plotly_chart(fig_chi_heat, use_container_width=True)

        if contingency_table.empty or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            st.warning("Contingency table is too small for Chi-Square test (e.g., one variable is constant).")
        elif (contingency_table < 5).any().any():
             st.warning("Some cells in the contingency table have expected frequencies < 5. Chi-Square test results might be less reliable. Consider Fisher's Exact Test for small tables.")
        
        try:
            chi2_stat, p_val_chi2, dof, expected_freq = stats.chi2_contingency(contingency_table)
            st.write(f"Chi-Square Statistic: {{chi2_stat:.4f}}, p-value: {{p_val_chi2:.4f}}, Degrees of Freedom: {{dof}}")
            if p_val_chi2 < alpha:
                st.write(f"**Conclusion:** The two categorical variables are likely **dependent** (p < {{alpha}}).")
            else:
                st.write(f"**Conclusion:** No significant evidence of dependence between the two categorical variables (p >= {{alpha}}).")
        except ValueError as ve:
            st.error(f"Error during Chi-Square test (often due to low frequencies or table shape): {{ve}}")

    else:
        st.write("Need at least two categorical columns for Chi-Square test.")
else:
    st.write("Not enough categorical columns for Chi-Square test.")

"""
        return code

    @staticmethod
    def anomaly_detection(df_placeholder_name="df"):
        """Generates Python code for statistical anomaly detection."""
        code = f"""
# Statistical Anomaly Detection
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA # For visualizing DBSCAN
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# df = st.session_state.df # Or however df is made available

st.write("### Anomaly Detection")
numeric_cols_ad = {df_placeholder_name}.select_dtypes(include=np.number).columns.tolist()

if not numeric_cols_ad:
    st.warning("No numeric columns found for anomaly detection.")
    st.stop()

# 1. Z-Score Method (Univariate)
st.write("#### 1. Z-Score Method (Univariate Outliers)")
z_threshold = 3 # Common threshold
st.write(f"Detecting outliers where absolute Z-score > {{z_threshold}}.")

z_score_summary = []
for col in numeric_cols_ad:
    data_col = {df_placeholder_name}[col].dropna()
    if len(data_col) == 0: continue
    
    z_scores = np.abs(stats.zscore(data_col))
    outliers_z = data_col[z_scores > z_threshold]
    z_score_summary.append({
        'Column': col,
        'Outliers Count (Z-score)': len(outliers_z),
        'Outliers %': (len(outliers_z) / len(data_col) * 100) if len(data_col) > 0 else 0
    })
    if len(outliers_z) > 0 and len(outliers_z) < 20: # Show some outliers if not too many
        st.write(f"Outliers in '{{col}}' (Z-score): {{outliers_z.tolist()[:5]}}{{'...' if len(outliers_z) > 5 else ''}}")

if z_score_summary:
    z_summary_df = pd.DataFrame(z_score_summary)
    st.dataframe(z_summary_df)
    
    # Box plots for columns with Z-score outliers
    cols_with_z_outliers = z_summary_df[z_summary_df['Outliers Count (Z-score)'] > 0]['Column'].tolist()
    if cols_with_z_outliers:
        st.write("##### Box Plots for Columns with Z-Score Outliers")
        num_plots_z = min(len(cols_with_z_outliers), 4)
        if num_plots_z > 0:
            fig_z_box = make_subplots(rows=(num_plots_z + 1) // 2, cols=2, subplot_titles=cols_with_z_outliers[:num_plots_z])
            for i, col_name in enumerate(cols_with_z_outliers[:num_plots_z]):
                row, col_idx = i // 2 + 1, i % 2 + 1
                fig_z_box.add_trace(go.Box(y={df_placeholder_name}[col_name], name=col_name, boxpoints='outliers'), row=row, col=col_idx)
            fig_z_box.update_layout(height=300 * ((num_plots_z + 1) // 2), showlegend=False)
            st.plotly_chart(fig_z_box, use_container_width=True)

# 2. IQR Method (Univariate)
st.write("#### 2. Interquartile Range (IQR) Method (Univariate Outliers)")
iqr_multiplier = 1.5 # Common multiplier
st.write(f"Detecting outliers outside Q1 - {{iqr_multiplier}}*IQR and Q3 + {{iqr_multiplier}}*IQR.")

iqr_summary = []
for col in numeric_cols_ad:
    data_col = {df_placeholder_name}[col].dropna()
    if len(data_col) == 0: continue

    Q1 = data_col.quantile(0.25)
    Q3 = data_col.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    outliers_iqr = data_col[(data_col < lower_bound) | (data_col > upper_bound)]
    iqr_summary.append({
        'Column': col,
        'Outliers Count (IQR)': len(outliers_iqr),
        'Outliers %': (len(outliers_iqr) / len(data_col) * 100) if len(data_col) > 0 else 0,
        'IQR Lower Bound': lower_bound,
        'IQR Upper Bound': upper_bound
    })
    if len(outliers_iqr) > 0 and len(outliers_iqr) < 20:
         st.write(f"Outliers in '{{col}}' (IQR): {{outliers_iqr.tolist()[:5]}}{{'...' if len(outliers_iqr) > 5 else ''}}")


if iqr_summary:
    iqr_summary_df = pd.DataFrame(iqr_summary)
    st.dataframe(iqr_summary_df)

    # Histograms with IQR bounds for columns with IQR outliers
    cols_with_iqr_outliers = iqr_summary_df[iqr_summary_df['Outliers Count (IQR)'] > 0]['Column'].tolist()
    if cols_with_iqr_outliers:
        st.write("##### Histograms with IQR Outlier Bounds")
        for col_name in cols_with_iqr_outliers[:min(3, len(cols_with_iqr_outliers))]: # Plot for first few
            fig_iqr_hist = px.histogram({df_placeholder_name}, x=col_name, marginal="box", title=f"Distribution of {{col_name}} with IQR Bounds")
            lb = iqr_summary_df[iqr_summary_df['Column'] == col_name]['IQR Lower Bound'].iloc[0]
            ub = iqr_summary_df[iqr_summary_df['Column'] == col_name]['IQR Upper Bound'].iloc[0]
            fig_iqr_hist.add_vline(x=lb, line_dash="dash", line_color="red", annotation_text="Lower Bound")
            fig_iqr_hist.add_vline(x=ub, line_dash="dash", line_color="red", annotation_text="Upper Bound")
            st.plotly_chart(fig_iqr_hist, use_container_width=True)


# 3. DBSCAN (Multivariate, if multiple numeric columns)
st.write("#### 3. DBSCAN for Multivariate Anomaly Detection")
if len(numeric_cols_ad) >= 2:
    st.write(f"Using features: {{', '.join(numeric_cols_ad)}} for DBSCAN.")
    
    # Prepare data: select numeric, drop NaNs, scale
    X_dbscan = {df_placeholder_name}[numeric_cols_ad].dropna()
    if X_dbscan.shape[0] < 5: # DBSCAN needs some points
        st.warning("Not enough data points after dropping NaNs for DBSCAN.")
    else:
        X_scaled = StandardScaler().fit_transform(X_dbscan)
        
        # DBSCAN parameters (these might need tuning or user input)
        eps_val = 0.5 # Default, common starting point
        min_samples_val = max(5, 2 * X_scaled.shape[1]) # Heuristic: 2 * num_dimensions
        st.write(f"DBSCAN parameters: eps={{eps_val}}, min_samples={{min_samples_val}}")

        try:
            dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
            clusters = dbscan.fit_predict(X_scaled)
            
            outliers_dbscan_count = np.sum(clusters == -1)
            st.write(f"DBSCAN identified {{outliers_dbscan_count}} multivariate outliers (labeled as -1).")
            
            if outliers_dbscan_count > 0:
                X_dbscan_results = X_dbscan.copy()
                X_dbscan_results['DBSCAN_Cluster'] = clusters
                
                st.write("Sample of data points with DBSCAN cluster labels (outliers are -1):")
                st.dataframe(X_dbscan_results.head())

                # Visualization (PCA if > 2 dimensions, else direct scatter)
                if X_scaled.shape[1] > 2:
                    st.write("Visualizing DBSCAN results using PCA (first 2 components):")
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'], index=X_dbscan_results.index)
                    pca_df['DBSCAN_Cluster'] = clusters
                    
                    fig_dbscan_pca = px.scatter(pca_df, x='PC1', y='PC2', color='DBSCAN_Cluster',
                                                color_continuous_scale=px.colors.sequential.Viridis, # Outliers (-1) might look odd here
                                                title='DBSCAN Clusters (PCA Reduced)')
                    # Highlight outliers specifically
                    outlier_trace = go.Scatter(x=pca_df[pca_df['DBSCAN_Cluster'] == -1]['PC1'],
                                               y=pca_df[pca_df['DBSCAN_Cluster'] == -1]['PC2'],
                                               mode='markers', marker=dict(color='red', size=8, symbol='x'), name='Outliers (-1)')
                    fig_dbscan_pca.add_trace(outlier_trace)
                    st.plotly_chart(fig_dbscan_pca, use_container_width=True)

                elif X_scaled.shape[1] == 2:
                    st.write(f"Visualizing DBSCAN results for {{numeric_cols_ad[0]}} vs {{numeric_cols_ad[1]}}:")
                    fig_dbscan_2d = px.scatter(X_dbscan_results, x=numeric_cols_ad[0], y=numeric_cols_ad[1],
                                               color='DBSCAN_Cluster',
                                               title='DBSCAN Clusters')
                    st.plotly_chart(fig_dbscan_2d, use_container_width=True)
        except Exception as e:
            st.error(f"Error during DBSCAN execution: {{e}}")
else:
    st.write("DBSCAN requires at least 2 numeric features for multivariate anomaly detection.")

"""
        return code
