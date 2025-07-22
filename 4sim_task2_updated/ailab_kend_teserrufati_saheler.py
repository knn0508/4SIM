# -*- coding: utf-8 -*-
"""
Kənd Təsərrüfatı və İqtisadi Sahələr üzrə ÜDM Analizi
"""

# Parameters
name = "AI"
surname = "Assistant"
sector = "Kənd Təsərrüfatı"
field = "İqtisadi Sahələr üzrə ÜDM"
start_year = 2024
end_year = 2028
generated_at = "2025-01-14 12:00:00"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import matplotlib
import os  # Add this import

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# Prophet model (alternative implementation)
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available, using alternative forecasting methods")

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from datetime import datetime, timedelta

# Set matplotlib to use available fonts
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)
# Enable Unicode support
plt.rcParams['axes.unicode_minus'] = False


def read_gdp_data():
    """Read and process GDP data from Excel file"""
    try:
        # Read the Excel file without header
        df_raw = pd.read_excel('Kend_Teserrufati_Saheleruzre.xlsx', header=None)

        # Create structured data
        data = {
            'Year': [],
            'Cəmi': [],
            'sənaye': [],
            'kənd təsərrüfatı, meşə təsərrüfatı və balıqçılıq': [],
            'tikinti': [],
            'nəqliyyat və rabitə': [],
            'xalis vergilər': [],
            'digər sahələr': []
        }

        # Extract million manat data (rows 5-28)
        for i in range(5, 29):  # 2000-2023 data
            try:
                row = df_raw.iloc[i]
                if pd.notna(row[1]):  # Check if year exists
                    year_str = str(row[1]).replace('*', '').strip()
                    year = int(year_str)
                    data['Year'].append(year)
                    data['Cəmi'].append(float(row[2]) if pd.notna(row[2]) else np.nan)
                    data['sənaye'].append(float(row[3]) if pd.notna(row[3]) else np.nan)
                    data['kənd təsərrüfatı, meşə təsərrüfatı və balıqçılıq'].append(
                        float(row[4]) if pd.notna(row[4]) else np.nan)
                    data['tikinti'].append(float(row[5]) if pd.notna(row[5]) else np.nan)
                    data['nəqliyyat və rabitə'].append(float(row[6]) if pd.notna(row[6]) else np.nan)
                    data['xalis vergilər'].append(float(row[7]) if pd.notna(row[7]) else np.nan)
                    data['digər sahələr'].append(float(row[8]) if pd.notna(row[8]) else np.nan)
            except (ValueError, TypeError):
                continue

        # Create DataFrame
        df = pd.DataFrame(data)
        df['Tarix'] = pd.to_datetime(df['Year'], format='%Y')
        df.set_index('Tarix', inplace=True)
        df.drop('Year', axis=1, inplace=True)

        return df

    except Exception as e:
        print(f"❌ Məlumat oxunma xətası: {e}")
        return None


def linear_trend_forecast(series, periods=5):
    """Linear trend forecasting as fallback"""
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
    try:
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=periods)
        conf_int = fitted_model.get_forecast(steps=periods).conf_int()
        return forecast.values, conf_int.values, fitted_model
    except:
        try:
            # Fallback to simple ARIMA
            model = ARIMA(series, order=(1, 1, 0))
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=periods)
            conf_int = fitted_model.get_forecast(steps=periods).conf_int()
            return forecast.values, conf_int.values, fitted_model
        except:
            # Final fallback to linear trend
            return linear_trend_forecast(series, periods)


def prophet_forecast(series, periods=5):
    """Prophet forecasting (if available)"""
    if not PROPHET_AVAILABLE:
        return linear_trend_forecast(series, periods)

    try:
        df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })

        model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
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
    # Create features (lagged values)
    n_lags = min(5, len(series) // 2)
    X, y = [], []

    for i in range(n_lags, len(series)):
        X.append(series.values[i - n_lags:i])
        y.append(series.values[i])

    X, y = np.array(X), np.array(y)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Generate forecasts
    forecasts = []
    last_values = series.values[-n_lags:]

    for _ in range(periods):
        pred = model.predict([last_values])[0]
        forecasts.append(pred)
        last_values = np.append(last_values[1:], pred)

    return np.array(forecasts), None, model


def perform_analysis():
    """Main analysis function"""
    df = read_gdp_data()
    if df is None:
        return None

    # Key sectors to analyze
    sectors = [
        'Cəmi',
        'sənaye',
        'kənd təsərrüfatı, meşə təsərrüfatı və balıqçılıq',
        'tikinti',
        'nəqliyyat və rabitə',
        'digər sahələr'
    ]

    results = {}

    for sector in sectors:
        series = df[sector]

        # ARIMA forecast
        arima_pred, arima_conf, arima_model = arima_forecast(series)

        # Prophet forecast
        prophet_pred, prophet_conf, prophet_model = prophet_forecast(series)

        # Random Forest forecast
        rf_pred, rf_conf, rf_model = random_forest_forecast(series)

        # Linear trend forecast
        linear_pred, linear_conf, linear_model = linear_trend_forecast(series)

        results[sector] = {
            'historical': series,
            'arima': {'forecast': arima_pred, 'conf_int': arima_conf, 'model': arima_model},
            'prophet': {'forecast': prophet_pred, 'conf_int': prophet_conf, 'model': prophet_model},
            'rf': {'forecast': rf_pred, 'conf_int': rf_conf, 'model': rf_model},
            'linear': {'forecast': linear_pred, 'conf_int': linear_conf, 'model': linear_model}
        }

    return results


def create_forecast_plot(sector, data, ax):
    """Create forecast visualization for a sector with custom methods for each"""
    historical = data['historical']

    # Plot historical data
    historical_line = ax.plot(historical.index, historical.values, 'o-', label='Tarixi məlumatlar',
                              color='blue', linewidth=2, markersize=4)

    # Add value labels for historical data (every 3rd point to avoid crowding)
    for i in range(0, len(historical), 3):
        ax.annotate(f'{int(historical.values[i])}',
                    xy=(historical.index[i], historical.values[i]),
                    xytext=(0, 10), textcoords='offset points',
                    fontsize=8, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    # Add annotation for the last historical point (same style as others)
    ax.annotate(f'{int(historical.values[-1])}',
                xy=(historical.index[-1], historical.values[-1]),
                xytext=(0, 10), textcoords='offset points',
                fontsize=8, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    # Future years
    future_years = pd.date_range(start='2024', periods=5, freq='Y')

    # Define which methods to keep for each sector
    methods_config = {
        'Cəmi': ['arima', 'linear'],
        'sənaye': ['arima', 'rf', 'prophet'],
        'kənd təsərrüfatı, meşə təsərrüfatı və balıqçılıq': ['arima', 'rf', 'linear'],
        'tikinti': ['arima', 'prophet'],
        'nəqliyyat və rabitə': ['arima', 'linear'],
        'digər sahələr': ['arima', 'prophet']
    }

    # Method display names and colors
    method_info = {
        'arima': {'name': 'ARIMA', 'color': 'red'},
        'prophet': {'name': 'Prophet', 'color': 'green'},
        'rf': {'name': 'Random Forest', 'color': 'orange'},
        'linear': {'name': 'Xətti Trend', 'color': 'purple'}
    }

    # Get methods to plot for this sector
    methods_to_plot = methods_config.get(sector, ['arima', 'prophet'])

    # Plot forecasts for selected methods
    for method in methods_to_plot:
        info = method_info[method]
        forecast = data[method]['forecast']

        # Plot forecast line (no value annotations)
        forecast_line = ax.plot(future_years, forecast, 'o--',
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
                elif conf_int.ndim == 1 and len(conf_int) >= 2:
                    margin = np.std(forecast) * 0.5
                    ax.fill_between(future_years, forecast - margin, forecast + margin,
                                    alpha=0.2, color=info['color'])
            except:
                pass  # Skip if error

    ax.set_title(f'{sector} - Zaman Seriyası Analizi və Proqnoz',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('İl', fontsize=12)
    ax.set_ylabel('Milyon Manat', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add trend line
    from scipy import stats
    years_numeric = np.arange(len(historical))
    slope, intercept, r_value, p_value, std_err = stats.linregress(years_numeric, historical.values)

    # Extend trend line to future
    all_years_numeric = np.arange(len(historical) + 5)
    trend_line = slope * all_years_numeric + intercept
    all_years = list(historical.index) + list(future_years)

    # Add explanations for predictions in each visual
    visual_explanations = {
        'Cəmi': "Göstərilən proqnozlara əsasən, ümumi ÜDM-də sabit artım trendi davam edəcək.",
        'sənaye': "Göstərilən proqnozlara əsasən, sənaye sektorunda volatillik davam edəcək.",
        'kənd təsərrüfatı, meşə təsərrüfatı və balıqçılıq': "Göstərilən proqnozlara əsasən, kənd təsərrüfatı sektorunda məhdud artım gözlənilir.",
        'tikinti': "Göstərilən proqnozlara əsasən, tikinti sektorunda güclü artım trendi davam edəcək.",
        'nəqliyyat və rabitə': "Göstərilən proqnozlara əsasən, nəqliyyat və rabitə sektorunda sabit artım gözlənilir.",
        'digər sahələr': "Göstərilən proqnozlara əsasən, digər sahələr sektorunda güclü artım trendi davam edəcək."
    }

    ax.plot(all_years, trend_line, '--', color='gray', alpha=0.7,
            label=f'Trend (R²={r_value ** 2:.3f})')
    ax.legend(fontsize=9)

    # Add explanation text below the plot
    ax.text(0.5, -0.25, visual_explanations.get(sector, ""),
            ha='center', va='top', transform=ax.transAxes,
            fontsize=10, style='italic', color='#555555')

    # Add a text box with key statistics
    stats_text = f"Son dəyər: {int(historical.values[-1])}\n"
    stats_text += f"Orta: {int(historical.mean())}\n"
    stats_text += f"Trend: {slope:.1f}/il"

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

def create_summary_statistics(results):
    """Create summary statistics table"""
    summary_data = []

    for sector, data in results.items():
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
                if hasattr(method_forecast, '__len__') and len(method_forecast) >= 5:
                    forecasts.append(method_forecast)
            except:
                continue

        if forecasts:
            avg_forecast_2024 = np.mean([f[0] for f in forecasts])
            avg_forecast_2025 = np.mean([f[1] for f in forecasts])
            avg_forecast_2026 = np.mean([f[2] for f in forecasts])
            avg_forecast_2027 = np.mean([f[3] for f in forecasts])
            avg_forecast_2028 = np.mean([f[4] for f in forecasts])
        else:
            # Fallback values if no forecasts available
            avg_forecast_2024 = historical.iloc[-1]
            avg_forecast_2025 = historical.iloc[-1]
            avg_forecast_2026 = historical.iloc[-1]
            avg_forecast_2027 = historical.iloc[-1]
            avg_forecast_2028 = historical.iloc[-1]

        summary_data.append({
            'Sektor': sector,
            'Orta (2000-2023)': f"{mean_val:.0f}",
            'Standart sapma': f"{std_val:.0f}",
            'İllik trend': f"{trend:.1f}",
            '2024 proqnozu': f"{avg_forecast_2024:.0f}",
            '2025 proqnozu': f"{avg_forecast_2025:.0f}",
            '2026 proqnozu': f"{avg_forecast_2026:.0f}",
            '2027 proqnozu': f"{avg_forecast_2027:.0f}",
            '2028 proqnozu': f"{avg_forecast_2028:.0f}"
        })

    return pd.DataFrame(summary_data)


def draw_summary_figure(pdf, total_gdp, sectors_2023, pie_labels, pie_sizes, growth_rates):
    """A4 səhifəsində 4 qrafiki estetik şəkildə çəkmək və PDF-ə əlavə etmək"""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    fig = plt.figure(figsize=(8.3, 11.7))  # A4 ölçüsü
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1.3, 1])  # Make pie chart section bigger
    fig.subplots_adjust(hspace=0.5)

    # Qrafik 1 – Ümumi ÜDM
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(total_gdp.index.year, total_gdp.values, color='#1976D2', linewidth=2, marker='o', markersize=4)
    ax1.set_title('Ümumi ÜDM (2000-2023)', fontsize=12, fontweight='bold', pad=8)
    ax1.set_xlabel('İl', fontsize=9)
    ax1.set_ylabel('Milyon Manat', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.annotate(f"{int(total_gdp.values[-1])}",
                 xy=(total_gdp.index.year[-1], total_gdp.values[-1]),
                 xytext=(5, 0), textcoords='offset points',
                 fontsize=8, fontweight='bold', color='#1976D2')
    ax1.tick_params(axis='both', labelsize=8)

    # Qrafik 2 – Barh: Sektorlara görə bölgü
    ax2 = fig.add_subplot(gs[1])
    bars = ax2.barh(list(sectors_2023.keys()), list(sectors_2023.values()), color='#43A047', alpha=0.85)

    ax2.set_title('2023-cü İldə Sektorlar üzrə Bölgü', fontsize=11, fontweight='bold', pad=6)
    ax2.set_xlabel('Milyon Manat', fontsize=9)
    ax2.tick_params(axis='both', labelsize=8)
    ax2.set_xlim(0, max(sectors_2023.values()) * 1.2)  # sağda boşluq

    for label in ax2.get_yticklabels():
        label.set_horizontalalignment('right')

    for bar in bars:
        ax2.text(bar.get_width() + max(sectors_2023.values()) * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 f"{int(bar.get_width())}", va='center', fontsize=7.5, fontweight='bold')

    # Qrafik 3 – Pie chart (Made bigger)
    ax3 = fig.add_subplot(gs[2])
    pie_colors = plt.cm.Set3(np.linspace(0, 1, len(pie_labels)))  # Better color scheme
    wedges, texts, autotexts = ax3.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%',
                                       startangle=140, colors=pie_colors,
                                       textprops={'fontsize': 10},
                                       pctdistance=0.85, radius=0.9)  # Increased radius from 0.65 to 0.9

    # Make percentage text more visible
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    # Make label text more visible
    for text in texts:
        text.set_fontsize(9)
        text.set_fontweight('bold')

    ax3.set_title('2023-cü il üzrə Sektorların Paylanması', fontsize=12, fontweight='bold', pad=10)

    # Qrafik 4 – Artım Dərəcəsi
    ax4 = fig.add_subplot(gs[3])
    growth_colors = ['#388E3C' if x > 0 else '#D32F2F' for x in growth_rates.values()]
    bars2 = ax4.bar(list(growth_rates.keys()), list(growth_rates.values()), color=growth_colors, alpha=0.85)
    ax4.set_title('2000-2023 Artım Dərəcəsi (%)', fontsize=12, fontweight='bold', pad=6)
    ax4.set_ylabel('Artım faizi', fontsize=9)
    ax4.tick_params(axis='x', labelsize=8, rotation=30)
    ax4.tick_params(axis='y', labelsize=8)
    for bar in bars2:
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.1f}%",
                 ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # PDF-ə əlavə et
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


def generate_pdf_report(results):
    """Generate comprehensive PDF report"""
    A4_SIZE = (8.3, 11.7)
    generated_pdf_path = f"{sector}_Sektoru_Hesabat_{end_year}.pdf"
    additional_pdf_path = "Kend_teserrufati_qeydler.pdf"
    merged_pdf_path = "Birləşdirilmiş_Hesabat_Agriculture.pdf"

    # First create the title page separately
    title_pdf_path = f"Title_{sector}_Sektoru_{end_year}.pdf"

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

    # Create the main content pages
    content_pdf_path = f"Content_{sector}_Sektoru_{end_year}.pdf"

    with PdfPages(content_pdf_path) as pdf:
        # Summary page with statistics
        total_gdp = results['Cəmi']['historical']
        sectors_2023 = {}
        for sector_name, data in results.items():
            if sector_name != 'Cəmi':
                sectors_2023[sector_name] = data['historical'].iloc[-1]

        pie_labels = list(sectors_2023.keys())
        pie_sizes = list(sectors_2023.values())

        growth_rates = {}
        for sector_name, data in results.items():
            if sector_name != 'Cəmi':
                hist = data['historical']
                growth_rate = ((hist.iloc[-1] - hist.iloc[0]) / hist.iloc[0]) * 100
                growth_rates[sector_name] = growth_rate

        draw_summary_figure(pdf, total_gdp, sectors_2023, pie_labels, pie_sizes, growth_rates)

        # Individual sector forecasts
        for sector_name, data in results.items():
            fig = plt.figure(figsize=A4_SIZE)
            ax = fig.add_axes([0.1, 0.5, 0.87, 0.45])
            create_forecast_plot(sector_name, data, ax)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

    # Try to merge PDFs with proper error handling
    try:
        from PyPDF2 import PdfMerger
        merger = PdfMerger()
        merger.append(title_pdf_path)  # Title page first

        # Check if Kend_teserrufati_qeydler.pdf exists and add it
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

    except ImportError:
        print("⚠️ PyPDF2 mövcud deyil, sadə PDF yaradılır")
        # If PyPDF2 is not available, just rename the content file
        os.rename(content_pdf_path, merged_pdf_path)
        print(f"✅ PDF yaradıldı: {merged_pdf_path}")

    except Exception as e:
        print(f"❌ PDF birləşdirmə xətası: {e}")
        # Just rename the content file as fallback
        try:
            os.rename(content_pdf_path, merged_pdf_path)
            print(f"✅ PDF yaradıldı (sadə versiya): {merged_pdf_path}")
        except:
            print(f"❌ PDF yaradılmasında xəta: {e}")


# Update the main execution
print("GDP analizi başladı...")
results = perform_analysis()
if results:
    generate_pdf_report(results)
    print("Hesabat tamamlandı: 'Birləşdirilmiş_Hesabat.pdf'")
else:
    print("Məlumat yüklənərkən xəta baş verdi!")