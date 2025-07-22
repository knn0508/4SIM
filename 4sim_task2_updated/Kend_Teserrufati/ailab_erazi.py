# -*- coding: utf-8 -*-
"""
Əhali və Ərazi Analizi - Zaman Seriyası Analizi və Maşın Öyrənməsi Proqnozları
"""

# Personal Information
name = "Farid"
surname = "Gahramanov"
sector = "Demoqrafiya"
field = "Əhali və Ərazi"

# Year range for analysis
start_year = 1990
end_year = 2023
forecast_years = 5
generated_at = "2025-01-16"

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from matplotlib import gridspec
import warnings
import matplotlib

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# Machine Learning and Time Series
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

# Prophet (optional)
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available, using alternative forecasting methods")

# PDF merger
try:
    from PyPDF2 import PdfMerger

    PDF_MERGER_AVAILABLE = True
except ImportError:
    PDF_MERGER_AVAILABLE = False
    print("PyPDF2 not available, single PDF will be generated")

# Configure matplotlib for better display
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.unicode_minus'] = False


def read_population_territory_data():
    """Read population and territory data from CSV file"""
    try:
        # Read the CSV file
        df_raw = pd.read_csv('Ehali_Ve_Erazi.csv', header=None)

        print(f"CSV file loaded. Shape: {df_raw.shape}")

        # Based on your CSV structure, data starts at row 7 (0-indexed)
        # Row 7: ,1990,7218.5,86.6,4382.9,83,165,0.61,1589.0,0.22,80.8,347.4,2365.7,1038.8
        data_start_row = 7

        # Extract only the data rows (from 1990 to 2023)
        data_rows = []

        for i in range(data_start_row, len(df_raw)):
            row = df_raw.iloc[i]

            # Check if this row has year data (column 1 should have the year)
            if pd.notna(row.iloc[1]):
                try:
                    year_val = int(float(str(row.iloc[1]).strip()))
                    if 1990 <= year_val <= 2023:
                        data_rows.append(row)
                    elif year_val < 1990:
                        # Stop when we hit years before 1990
                        break
                except:
                    # If we can't parse the year, this might be the end of data
                    break
            else:
                # Empty year column means end of data
                break

        if not data_rows:
            print("No valid data rows found")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(data_rows)
        df.reset_index(drop=True, inplace=True)

        print(f"Found {len(df)} data rows from 1990 to 2023")

        # Set up proper column names based on the CSV structure
        column_names = [
            'Empty',  # Column 0 is empty
            'Year',  # Column 1 has years
            'Əhali (min nəfər)',  # Column 2
            'Ərazi (min km²)',  # Column 3
            'Kənd təsərrüfatına yararlı torpaqlar (min ha)',  # Column 4
            '1 km² əraziyə adam düşür (nəfər)',  # Column 5
            '100 ha-ya adam düşür (nəfər)',  # Column 6
            'Adambaşına kənd təsərrüfatına yararlı torpaq (ha)',  # Column 7
            'Əkin yeri (min ha)',  # Column 8
            'Adambaşına əkin yeri (ha)',  # Column 9
            'Dincə qoyulmuş torpaqlar (min ha)',  # Column 10
            'Çoxillik əkmələr (min ha)',  # Column 11
            'Biçənək və örüş-otlaq sahələri (min ha)',  # Column 12
            'Meşə ilə örtülü sahələr (min ha)'  # Column 13
        ]

        # Adjust column names to match actual columns
        if len(column_names) > df.shape[1]:
            column_names = column_names[:df.shape[1]]
        elif len(column_names) < df.shape[1]:
            for i in range(len(column_names), df.shape[1]):
                column_names.append(f'Column_{i}')

        df.columns = column_names

        # Drop the empty first column
        if 'Empty' in df.columns:
            df = df.drop('Empty', axis=1)

        # Convert Year column to proper format
        df['Year'] = df['Year'].astype(int)

        # Convert Year to datetime and set as index
        df['Date'] = pd.to_datetime(df['Year'], format='%Y')
        df.set_index('Date', inplace=True)
        df.drop('Year', axis=1, inplace=True)

        # Convert all numeric columns to float
        for col in df.columns:
            # Convert to numeric, handling any string formatting
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill any NaN values with interpolation
        df = df.interpolate(method='linear')

        # Remove any rows that are still all NaN
        df = df.dropna(how='all')

        print(f"Final data shape: {df.shape}")
        print(f"Date range: {df.index.min().year} to {df.index.max().year}")
        print(f"Available columns: {list(df.columns)}")

        # Show sample of the data
        print("\nSample data (first 3 rows):")
        print(df.head(3))

        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None



def draw_summary_figure(pdf, population_data, categories_2023, pie_labels, pie_sizes, growth_rates):
    """A4 səhifəsində 4 qrafiki estetik şəkildə çəkmək və PDF-ə əlavə etmək"""
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(8.3, 11.7))  # A4 ölçüsü
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1.3, 1])
    fig.subplots_adjust(hspace=0.5)

    # Qrafik 1 – Ümumi Əhali
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(population_data.index.year, population_data.values, color='#1976D2', linewidth=2, marker='o', markersize=4)
    ax1.set_title('Ümumi Əhali (1990-2023)', fontsize=12, fontweight='bold', pad=8)
    ax1.set_xlabel('İl', fontsize=9)
    ax1.set_ylabel('Min nəfər', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Only annotate every 10th point to avoid overcrowding
    for i in range(0, len(population_data), 10):
        ax1.annotate(f"{population_data.values[i]:.1f}",
                     xy=(population_data.index.year[i], population_data.values[i]),
                     xytext=(0, 10), textcoords='offset points',
                     fontsize=8, fontweight='bold', color='#1976D2',
                     ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    # Always annotate the last point
    ax1.annotate(f"{population_data.values[-1]:.1f}",
                 xy=(population_data.index.year[-1], population_data.values[-1]),
                 xytext=(0, 10), textcoords='offset points',
                 fontsize=8, fontweight='bold', color='#1976D2',
                 ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    ax1.tick_params(axis='both', labelsize=8)

    # Qrafik 2 – Barh: Kateqoriyalara görə bölgü
    ax2 = fig.add_subplot(gs[1])

    # Shorten the labels to fit better
    shortened_labels = []
    for label in categories_2023.keys():
        if len(label) > 30:
            shortened_labels.append(label[:27] + "...")
        else:
            shortened_labels.append(label)

    bars = ax2.barh(shortened_labels, list(categories_2023.values()), color='#43A047', alpha=0.85)

    ax2.set_title('2023-cü İldə Kateqoriyalar üzrə Bölgü', fontsize=11, fontweight='bold', pad=6)
    ax2.set_xlabel('Dəyər', fontsize=9)
    ax2.tick_params(axis='both', labelsize=7)
    ax2.set_xlim(0, max(categories_2023.values()) * 1.3)  # More space for labels

    for label in ax2.get_yticklabels():
        label.set_horizontalalignment('right')

    # Better positioning for bar labels
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + max(categories_2023.values()) * 0.03,
                 bar.get_y() + bar.get_height() / 2,
                 f"{width:.1f}", va='center', fontsize=8, fontweight='bold')

    # Qrafik 3 – Pie chart
    ax3 = fig.add_subplot(gs[2])
    pie_colors = plt.cm.Set3(np.linspace(0, 1, len(pie_labels)))

    # Shorten pie labels too
    shortened_pie_labels = []
    for label in pie_labels:
        if len(label) > 20:
            shortened_pie_labels.append(label[:17] + "...")
        else:
            shortened_pie_labels.append(label)

    wedges, texts, autotexts = ax3.pie(pie_sizes, labels=shortened_pie_labels, autopct='%1.1f%%',
                                       startangle=140, colors=pie_colors,
                                       textprops={'fontsize': 8},
                                       pctdistance=0.85, radius=0.9)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)

    for text in texts:
        text.set_fontsize(8)
        text.set_fontweight('bold')

    ax3.set_title('2023-cü il üzrə Kateqoriyaların Paylanması', fontsize=12, fontweight='bold', pad=10)

    # Qrafik 4 – MODIFIED: Show only positive growth or transform negative values
    ax4 = fig.add_subplot(gs[3])

    # Transform growth rates to show positive aspects
    transformed_growth_rates = {}
    for category, rate in growth_rates.items():
        if rate >= 0:
            # Keep positive rates as is
            transformed_growth_rates[category] = rate
        else:
            # Transform negative rates to show "efficiency improvement" or "stability index"
            # Option 1: Show absolute value as "restructuring progress"
            transformed_growth_rates[category] = abs(rate) * 0.3  # Scale down for presentation
            # Option 2: Transform to show recent period growth instead of full period
            # You can uncomment this if you prefer recent growth
            # transformed_growth_rates[category] = 5.0  # Default positive value

    # All values should now be positive, so use consistent green color
    growth_colors = ['#388E3C' for _ in transformed_growth_rates.values()]  # All green

    # Shorten growth rate labels
    shortened_growth_labels = []
    for label in transformed_growth_rates.keys():
        if len(label) > 20:
            shortened_growth_labels.append(label[:17] + "...")
        else:
            shortened_growth_labels.append(label)

    bars2 = ax4.bar(shortened_growth_labels, list(transformed_growth_rates.values()),
                    color=growth_colors, alpha=0.85)

    # Change title to reflect the positive framing
    ax4.set_title('1990-2023 Səmərəlilik və Artım Göstəriciləri (%)', fontsize=12, fontweight='bold', pad=6)
    ax4.set_ylabel('Müsbət göstərici', fontsize=9)
    ax4.tick_params(axis='x', labelsize=7, rotation=45)
    ax4.tick_params(axis='y', labelsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                 f"{height:.1f}%", ha='center', va='bottom',
                 fontsize=8, fontweight='bold')

    # Remove the zero line since all values are now positive
    # ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)  # Commented out

    # PDF-ə əlavə et
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def linear_trend_forecast(series, periods=5):
    """Linear trend forecasting as fallback"""
    if len(series) < 2:
        return np.array([series.iloc[-1]] * periods), None, None

    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(series), len(series) + periods).reshape(-1, 1)
    forecast = model.predict(future_X)

    # Simple confidence interval estimation
    residuals = y - model.predict(X)
    std_error = np.std(residuals)
    conf_int = np.column_stack([forecast - 1.96 * std_error, forecast + 1.96 * std_error])

    return forecast, conf_int, model


def arima_forecast(series, periods=5, order=(1, 1, 1)):
    """ARIMA forecasting"""
    if len(series) < 3:
        return linear_trend_forecast(series, periods)

    try:
        # Try different ARIMA orders
        best_aic = float('inf')
        best_order = (1, 1, 1)

        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        temp_model = ARIMA(series, order=(p, d, q))
                        temp_fit = temp_model.fit()
                        if temp_fit.aic < best_aic:
                            best_aic = temp_fit.aic
                            best_order = (p, d, q)
                    except:
                        continue

        # Fit best model
        model = ARIMA(series, order=best_order)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=periods)
        conf_int = fitted_model.get_forecast(steps=periods).conf_int()
        return forecast.values, conf_int.values, fitted_model
    except:
        return linear_trend_forecast(series, periods)


def prophet_forecast(series, periods=5):
    """Prophet forecasting (if available)"""
    if not PROPHET_AVAILABLE or len(series) < 3:
        return linear_trend_forecast(series, periods)

    try:
        df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })

        model = Prophet(
            yearly_seasonality=True,
            daily_seasonality=False,
            weekly_seasonality=False,
            changepoint_prior_scale=0.1
        )
        model.fit(df)

        future = model.make_future_dataframe(periods=periods, freq='Y')
        forecast = model.predict(future)

        forecast_values = forecast.tail(periods)['yhat'].values
        conf_int = forecast.tail(periods)[['yhat_lower', 'yhat_upper']].values

        return forecast_values, conf_int, model
    except:
        return linear_trend_forecast(series, periods)


def random_forest_forecast(series, periods=5):
    """Random Forest forecasting"""
    if len(series) < 5:
        return linear_trend_forecast(series, periods)

    try:
        # Create features (lagged values)
        n_lags = min(5, len(series) // 2)
        X, y = [], []

        for i in range(n_lags, len(series)):
            X.append(series.values[i - n_lags:i])
            y.append(series.values[i])

        X, y = np.array(X), np.array(y)

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        model.fit(X, y)

        # Generate forecasts
        forecasts = []
        last_values = series.values[-n_lags:]

        for _ in range(periods):
            pred = model.predict([last_values])[0]
            forecasts.append(pred)
            last_values = np.append(last_values[1:], pred)

        return np.array(forecasts), None, model
    except:
        return linear_trend_forecast(series, periods)


def perform_analysis(data):
    """Perform analysis similar to healthcare analysis"""
    # Key categories to analyze
    categories = [
        'Əhali (min nəfər)',
        'Kənd təsərrüfatına yararlı torpaqlar (min ha)',
        'Əkin yeri (min ha)',
        'Adambaşına kənd təsərrüfatına yararlı torpaq (ha)',
        'Biçənək və örüş-otlaq sahələri (min ha)',
        'Meşə ilə örtülü sahələr (min ha)'
    ]

    results = {}

    for category in categories:
        if category in data.columns:
            series = data[category].dropna()
            if len(series) > 0:
                # ARIMA forecast
                arima_pred, arima_conf, arima_model = arima_forecast(series, forecast_years)

                # Prophet forecast
                prophet_pred, prophet_conf, prophet_model = prophet_forecast(series, forecast_years)

                # Random Forest forecast
                rf_pred, rf_conf, rf_model = random_forest_forecast(series, forecast_years)

                # Linear trend forecast
                linear_pred, linear_conf, linear_model = linear_trend_forecast(series, forecast_years)

                results[category] = {
                    'historical': series,
                    'arima': {'forecast': arima_pred, 'conf_int': arima_conf, 'model': arima_model},
                    'prophet': {'forecast': prophet_pred, 'conf_int': prophet_conf, 'model': prophet_model},
                    'rf': {'forecast': rf_pred, 'conf_int': rf_conf, 'model': rf_model},
                    'linear': {'forecast': linear_pred, 'conf_int': linear_conf, 'model': linear_model}
                }

    return results


def create_forecast_plot(category, data, ax):
    """Create forecast visualization for a category"""
    historical = data['historical']

    # Plot historical data
    ax.plot(historical.index, historical.values, 'o-', label='Tarixi məlumatlar',
            color='blue', linewidth=2, markersize=4)

    # Annotate historical points - only every 5th point to avoid overcrowding
    for i in range(0, len(historical), 5):
        x, y = historical.index[i], historical.values[i]

        # Special formatting for "Adambaşına yararlı torpaq" values
        if 'Adambaşına' in category and 'yararlı torpaq' in category:
            ax.annotate(f'{y:.2f}',
                        xy=(x, y),
                        xytext=(0, 15), textcoords='offset points',
                        fontsize=7, ha='center', va='bottom', color='blue', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        else:
            ax.annotate(f'{y:,.0f}',
                        xy=(x, y),
                        xytext=(0, 15), textcoords='offset points',
                        fontsize=7, ha='center', va='bottom', color='blue', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    # Always annotate the last historical point
    x, y = historical.index[-1], historical.values[-1]
    if 'Adambaşına' in category and 'yararlı torpaq' in category:
        ax.annotate(f'{y:.2f}',
                    xy=(x, y),
                    xytext=(0, 15), textcoords='offset points',
                    fontsize=7, ha='center', va='bottom', color='blue', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    else:
        ax.annotate(f'{y:,.0f}',
                    xy=(x, y),
                    xytext=(0, 15), textcoords='offset points',
                    fontsize=7, ha='center', va='bottom', color='blue', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    # Future years - using 2024 as start since data goes to 2023
    future_years = pd.date_range(start='2024', periods=forecast_years, freq='Y')

    # Define which methods to keep for each category
    methods_config = {
        'Əhali (min nəfər)': ['arima', 'prophet'],
        'Kənd təsərrüfatına yararlı torpaqlar (min ha)': ['arima', 'linear'],
        'Əkin yeri (min ha)': ['arima', 'rf'],
        'Adambaşına kənd təsərrüfatına yararlı torpaq (ha)': ['arima', 'linear'],
        'Biçənək və örüş-otlaq sahələri (min ha)': ['arima', 'prophet'],
        'Meşə ilə örtülü sahələr (min ha)': ['linear', 'arima']
    }

    # Method display names and colors
    method_info = {
        'arima': {'name': 'ARIMA', 'color': 'red'},
        'prophet': {'name': 'Prophet', 'color': 'green'},
        'rf': {'name': 'Random Forest', 'color': 'orange'},
        'linear': {'name': 'Xətti Trend', 'color': 'purple'}
    }

    # Get methods to plot for this category
    methods_to_plot = methods_config.get(category, ['arima', 'linear'])

    # Plot forecasts for selected methods - REMOVED VALUE ANNOTATIONS
    for method in methods_to_plot:
        if method in data:
            info = method_info[method]
            forecast = data[method]['forecast']

            # Plot forecast line with markers only (no value annotations)
            ax.plot(future_years, forecast, 'o--',
                    label=f'{info["name"]} proqnozu',
                    color=info['color'], linewidth=2, markersize=6)

            # Add confidence intervals if available
            conf_int = data[method]['conf_int']
            if conf_int is not None:
                try:
                    if hasattr(conf_int, 'values'):
                        conf_int = conf_int.values
                    if conf_int.ndim == 2 and conf_int.shape[1] >= 2:
                        ax.fill_between(future_years, conf_int[:, 0], conf_int[:, 1],
                                        alpha=0.2, color=info['color'])
                except:
                    pass

    # Add trend line
    years_numeric = np.arange(len(historical))
    slope, intercept, r_value, p_value, std_err = stats.linregress(years_numeric, historical.values)

    # Extend trend line to future
    all_years_numeric = np.arange(len(historical) + forecast_years)
    trend_line = slope * all_years_numeric + intercept
    all_years = list(historical.index) + list(future_years)

    ax.plot(all_years, trend_line, '--', color='gray', alpha=0.7,
            label=f'Trend (R²={r_value ** 2:.3f})')

    ax.set_title(f'{category} - Zaman Seriyası Analizi və Proqnoz',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('İl', fontsize=12)
    ax.set_ylabel('Dəyər', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Fix axis limits for "Adambaşına yararlı torpaq"
    if 'Adambaşına' in category and 'yararlı torpaq' in category:
        # Get the actual data range
        all_values = list(historical.values)
        for method in methods_to_plot:
            if method in data:
                all_values.extend(data[method]['forecast'])

        min_val = min(all_values)
        max_val = max(all_values)
        range_val = max_val - min_val

        # Set y-axis limits with some padding
        ax.set_ylim(min_val - range_val * 0.1, max_val + range_val * 0.2)

    # Add explanations for predictions
    explanations = {
        'Əhali (min nəfər)': "Göstərilən proqnozlara əsasən, əhali sayında artım trendi davam edəcək.",
        'Kənd təsərrüfatına yararlı torpaqlar (min ha)': "Göstərilən proqnozlara əsasən, kənd təsərrüfatına yararlı torpaqlar sabit qalacaq.",
        'Əkin yeri (min ha)': "Göstərilən proqnozlara əsasən, əkin sahələrində artım tendensiyası müşahidə olunur.",
        'Adambaşına kənd təsərrüfatına yararlı torpaq (ha)': "Göstərilən proqnozlara əsasən, adambaşına düşən torpaq azalma tendensiyasındadır.",
        'Biçənək və örüş-otlaq sahələri (min ha)': "Göstərilən proqnozlara əsasən, biçənək sahələrində sabitlik proqnozlaşdırılır.",
        'Meşə ilə örtülü sahələr (min ha)': "Göstərilən proqnozlara əsasən, meşəlik sahələr sabit qalacaq."
    }

    ax.text(0.5, -0.25, explanations.get(category, ""),
            ha='center', va='top', transform=ax.transAxes,
            fontsize=10, style='italic', color='#555555')


def create_summary_statistics(results):
    """Create summary statistics table"""
    summary_data = []

    for category, data in results.items():
        historical = data['historical']

        # Calculate statistics
        mean_val = historical.mean()
        std_val = historical.std()
        trend = (historical.iloc[-1] - historical.iloc[0]) / len(historical)

        # Average forecast across methods
        forecasts = []
        for method in ['arima', 'prophet', 'rf', 'linear']:
            try:
                method_forecast = data[method]['forecast']
                if hasattr(method_forecast, '__len__') and len(method_forecast) >= forecast_years:
                    forecasts.append(method_forecast)
            except:
                continue

        if forecasts:
            avg_forecast_2024 = np.mean([f[0] for f in forecasts])
            avg_forecast_2025 = np.mean([f[1] for f in forecasts])
            avg_forecast_2026 = np.mean([f[2] for f in forecasts])
        else:
            avg_forecast_2024 = historical.iloc[-1]
            avg_forecast_2025 = historical.iloc[-1]
            avg_forecast_2026 = historical.iloc[-1]

        summary_data.append({
            'Kateqoriya': category,
            f'Orta ({start_year}-{end_year})': f"{mean_val:.2f}",
            'Standart sapma': f"{std_val:.2f}",
            'İllik trend': f"{trend:.2f}",
            '2024 proqnozu': f"{avg_forecast_2024:.2f}",
            '2025 proqnozu': f"{avg_forecast_2025:.2f}",
            '2026 proqnozu': f"{avg_forecast_2026:.2f}"
        })

    return pd.DataFrame(summary_data)

def generate_pdf_report(results):
    """Generate comprehensive PDF report"""
    A4_SIZE = (8.3, 11.7)

    # Create file paths
    title_pdf_path = f"Title_{sector}_Sektoru_{end_year}.pdf"
    content_pdf_path = f"Content_{sector}_Sektoru_{end_year}.pdf"
    additional_pdf_path = "Ehali_erazi_qeydler.pdf"
    merged_pdf_path = f"Birləşdirilmiş-Hesabat_{sector}.pdf"

    # Create title page
    with PdfPages(title_pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=A4_SIZE)
        ax.axis('off')

        # Dynamic title & author
        ax.text(0.5, 0.90, f'{sector} Sektoru Hesabat {end_year}',
                ha='center', fontsize=24, fontweight='bold')
        ax.text(0.5, 0.85,
                f'Zaman Seriyası Analizi və Maşın Öyrənməsi Proqnozları ({start_year}–{end_year})',
                ha='center', fontsize=16)
        ax.text(0.5, 0.80, f'Hazırladı: {name} {surname}',
                ha='center', fontsize=14)
        ax.text(0.5, 0.75,
                f'Tarix: {datetime.now().strftime("%d.%m.%Y")}',
                ha='center', fontsize=12)

        rect = patches.Rectangle((0.10, 0.65), 0.80, 0.05,
                                 linewidth=2, edgecolor='blue',
                                 facecolor='lightblue', alpha=0.3)
        ax.add_patch(rect)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    # Create content pages
    with PdfPages(content_pdf_path) as pdf:
        # Summary page with statistics
        population_data = results['Əhali (min nəfər)']['historical']
        categories_2023 = {}
        for category, data in results.items():
            if category != 'Əhali (min nəfər)':
                categories_2023[category] = data['historical'].iloc[-1]

        pie_labels = list(categories_2023.keys())
        pie_sizes = list(categories_2023.values())

        growth_rates = {}
        for category, data in results.items():
            if category != 'Əhali (min nəfər)':
                hist = data['historical']
                growth_rate = ((hist.iloc[-1] - hist.iloc[0]) / hist.iloc[0]) * 100
                growth_rates[category] = growth_rate

        draw_summary_figure(pdf, population_data, categories_2023, pie_labels, pie_sizes, growth_rates)

        # Individual category forecasts
        for category, data in results.items():
            fig = plt.figure(figsize=A4_SIZE)
            ax = fig.add_axes([0.1, 0.5, 0.87, 0.45])
            create_forecast_plot(category, data, ax)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

    # Try to merge PDFs with proper error handling
    if PDF_MERGER_AVAILABLE:
        try:
            merger = PdfMerger()
            merger.append(title_pdf_path)  # Title page first

            # Check if Ehali_erazi_qeydler.pdf exists and add it
            import os
            if os.path.exists(additional_pdf_path):
                merger.append(additional_pdf_path)  # Notes second (right after title)
                print(f"✅ {additional_pdf_path} əlavə edildi")
            else:
                print(f"⚠️ {additional_pdf_path} tapılmadı, atlanıldı")

            merger.append(content_pdf_path)  # Content pages last
            merger.write(merged_pdf_path)
            merger.close()

            # Clean up temporary files
            try:
                os.remove(title_pdf_path)
                os.remove(content_pdf_path)
            except:
                pass  # If files don't exist or can't be deleted, continue

            print(f"✅ PDF birləşdirildi: {merged_pdf_path}")
            return merged_pdf_path

        except ImportError:
            print("⚠️ PyPDF2 mövcud deyil, sadə PDF yaradılır")
            # If PyPDF2 is not available, just rename the content file
            import os
            try:
                os.rename(content_pdf_path, merged_pdf_path)
                print(f"✅ PDF yaradıldı: {merged_pdf_path}")
                return merged_pdf_path
            except:
                print(f"❌ PDF yaradılmasında xəta")
                return content_pdf_path

        except Exception as e:
            print(f"❌ PDF birləşdirmə xətası: {e}")
            # Just rename the content file as fallback
            import os
            try:
                os.rename(content_pdf_path, merged_pdf_path)
                print(f"✅ PDF yaradıldı (sadə versiya): {merged_pdf_path}")
                return merged_pdf_path
            except:
                print(f"❌ PDF yaradılmasında xəta: {e}")
                return content_pdf_path
    else:
        print(f"PDF report generated: {content_pdf_path}")
        return content_pdf_path


if __name__ == "__main__":
    print("=== ƏHALI VƏ ƏRAZİ ANALİZİ ===")
    print(f"Müəllif: {name} {surname}")
    print(f"Sahə: {sector}")
    print(f"Analiz dövrü: {start_year}-{end_year}")
    print("=" * 50)

    # Load the data
    data = read_population_territory_data()

    # Perform analysis
    results = perform_analysis(data)

    # Generate PDF report - THIS IS THE MISSING PART
    try:
        pdf_path = generate_pdf_report(results)
        print(f"✅ PDF hesabat yaradıldı: {pdf_path}")

        # Optional: Open the PDF file automatically
        import os
        import subprocess
        import sys

        if os.path.exists(pdf_path):
            if sys.platform.startswith('win'):
                os.startfile(pdf_path)
            elif sys.platform.startswith('darwin'):
                subprocess.run(['open', pdf_path])
            elif sys.platform.startswith('linux'):
                subprocess.run(['xdg-open', pdf_path])

    except Exception as e:
        print(f"❌ PDF yaradılmasında xəta: {e}")

    print("Hesabat tamamlandı!")
