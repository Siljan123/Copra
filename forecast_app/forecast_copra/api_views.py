import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from django.http import JsonResponse
from .models import ForecastLog
import io, base64
from .models import TrainingData, TrainedModel, ForecastLog
from .utils.arimax_model import ARIMAXModel


def historical_trend_api(request):
    """
    API endpoint that returns all data shown in the historical_trend view.
    Query param ?graph=true includes the chart as a base64 string.
    Now also returns pure data series for actual prices, test-set predictions,
    and user forecast logs.
    """
    include_graph = request.GET.get('graph', 'false').lower() == 'true'

    # ------------------------------------------------------------------
    # 1. Load all actual price data
    # ------------------------------------------------------------------
    data_qs = TrainingData.objects.all().order_by('date')
    if not data_qs.exists():
        return JsonResponse({'no_data': True}, status=200)

    df = pd.DataFrame(list(data_qs.values('date', 'farmgate_price')))
    df['date'] = pd.to_datetime(df['date'])
    df['farmgate_price'] = pd.to_numeric(df['farmgate_price'], errors='coerce').astype(float)
    df = df.sort_values('date').reset_index(drop=True)

    # Pure data: actual series (all dates)
    actual_series = [
        {'date': row['date'].strftime('%Y-%m-%d'), 'price': round(float(row['farmgate_price']), 2)}
        for _, row in df.iterrows()
    ]

    # ------------------------------------------------------------------
    # 2. Active model – test set actual vs predicted
    # ------------------------------------------------------------------
    eval_rows = []
    eval_dates = pd.DatetimeIndex([])
    eval_actual = []
    eval_preds = []
    active_model = None
    training_cutoff_date = None
    test_set_predicted_series = []   # pure data for predicted line

    try:
        active_model = TrainedModel.objects.filter(
            is_active=True,
            plot_actual__isnull=False,
            plot_preds__isnull=False,
        ).order_by('-training_date').first()

        if active_model:
            raw_actual = list(active_model.plot_actual)
            raw_preds = list(active_model.plot_preds)
            n = min(len(raw_actual), len(df))

            tail_slice = df.iloc[-n:].reset_index(drop=True)
            eval_actual = raw_actual[:n]
            eval_preds = raw_preds[:n]
            eval_dates = pd.to_datetime(tail_slice['date'].values)

            if len(eval_dates) > 0:
                # Build existing eval_rows (for backward compatibility)
                eval_rows = [
                    {
                        'date': pd.Timestamp(d).strftime('%b %d, %Y'),
                        'actual': round(float(a), 2),
                        'predicted': round(float(p), 2),
                        'error': round(abs(float(a) - float(p)), 2),
                        'error_pct': round(
                            abs(float(a) - float(p)) / (abs(float(a)) + 1e-10) * 100, 2
                        ),
                    }
                    for d, a, p in zip(eval_dates, eval_actual, eval_preds)
                ]
                # NEW: pure data series for test-set predicted line
                test_set_predicted_series = [
                    {'date': pd.Timestamp(d).strftime('%Y-%m-%d'), 'predicted_price': round(float(p), 2)}
                    for d, p in zip(eval_dates, eval_preds)
                ]
                training_cutoff_date = eval_dates[-1]

            if training_cutoff_date is None:
                training_cutoff_date = df['date'].max()
    except Exception as e:
        print(f"[historical_trend_api] eval error: {e}")

    # Fallback for cutoff date
    if training_cutoff_date is None:
        active_model = TrainedModel.objects.filter(is_active=True).order_by('-training_date').first()
        if active_model:
            training_cutoff_date = active_model.training_date.date()
        else:
            training_cutoff_date = df['date'].max()

    # ------------------------------------------------------------------
    # 3. Admin new data (recent window)
    # ------------------------------------------------------------------
    admin_rows = []
    admin_df = pd.DataFrame()
    admin_window_days = 5
    admin_window_start = None
    if training_cutoff_date is not None:
        admin_window_start = training_cutoff_date - timedelta(days=admin_window_days)

    if admin_window_start is not None:
        after_qs = TrainingData.objects.filter(date__gte=admin_window_start).order_by('date')
    else:
        after_qs = TrainingData.objects.all().order_by('date')

    if after_qs.exists():
        admin_df = pd.DataFrame(list(after_qs.values('date', 'farmgate_price')))
        admin_df['date'] = pd.to_datetime(admin_df['date'])
        admin_df['farmgate_price'] = pd.to_numeric(admin_df['farmgate_price'], errors='coerce').astype(float)
        admin_df = admin_df.sort_values('date').reset_index(drop=True)

        admin_rows = [
            {'date': row['date'].strftime('%b %d, %Y'), 'price': round(float(row['farmgate_price']), 2)}
            for _, row in admin_df.iterrows()
        ]

    # ------------------------------------------------------------------
    # 4. User forecast logs – pure data series for the predicted line
    # ------------------------------------------------------------------
    log_rows = []
    log_df = pd.DataFrame()
    forecast_log_series = []   # pure data for forecast log line
    logs = ForecastLog.objects.all().order_by('created_at')
    if logs.exists():
        log_df = pd.DataFrame(list(logs.values('created_at', 'price_predicted', 'forecast_horizon')))
        log_df['created_at'] = pd.to_datetime(log_df['created_at']).dt.tz_localize(None)
        log_df['target_date'] = log_df.apply(
            lambda x: x['created_at'] + timedelta(days=int(x['forecast_horizon'] or 0)),
            axis=1,
        )
        log_df = log_df.sort_values('target_date').reset_index(drop=True)

        log_rows = [
            {'date': row['target_date'].strftime('%b %d, %Y'), 'predicted': round(float(row['price_predicted']), 2)}
            for _, row in log_df.iterrows()
        ]
        # NEW: pure data series for forecast logs (target date vs predicted)
        forecast_log_series = [
            {'date': row['target_date'].strftime('%Y-%m-%d'), 'predicted_price': round(float(row['price_predicted']), 2)}
            for _, row in log_df.iterrows()
        ]

    # ------------------------------------------------------------------
    # 5. Admin vs forecast comparison (exact date matches)
    # ------------------------------------------------------------------
    admin_forecast_rows = []
    if not admin_df.empty and not log_df.empty:
        merged = pd.merge(
            admin_df,
            log_df[['target_date', 'price_predicted']],
            left_on='date',
            right_on='target_date',
            how='inner'
        )
        if not merged.empty:
            admin_forecast_rows = [
                {
                    'date': row['date'].strftime('%b %d, %Y'),
                    'actual': round(float(row['farmgate_price']), 2),
                    'forecast': round(float(row['price_predicted']), 2),
                    'diff': round(float(row['farmgate_price']) - float(row['price_predicted']), 2),
                    'diff_pct': round(
                        abs(float(row['farmgate_price']) - float(row['price_predicted']))
                        / (abs(float(row['farmgate_price'])) + 1e-10) * 100, 2
                    ),
                }
                for _, row in merged.iterrows()
            ]

    # ------------------------------------------------------------------
    # 6. Build the chart (only if requested) – unchanged
    # ------------------------------------------------------------------
    graph_base64 = None
    if include_graph:
        plt.close('all')
        all_chart_dates = list(eval_dates) + list(admin_df['date']) + list(log_df['target_date'])
        all_chart_dates = [d for d in all_chart_dates if pd.notnull(d)]
        n_dates = max(len(all_chart_dates), 10)
        fig_width = max(14, n_dates * 0.55)

        fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=140)
        bg = '#0f172a'
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        # test-set actual
        if len(eval_dates) > 0:
            d_pts = np.ravel(eval_dates)
            a_pts = np.ravel(eval_actual)
            ax.plot(d_pts, a_pts, color='#10b981', lw=2, marker='o', markersize=4,
                    markerfacecolor='#065f46', markeredgecolor='#10b981',
                    label='Actual Price (Test Set)', zorder=4)
            ax.fill_between(d_pts, a_pts, color='#10b981', alpha=0.07)

        # test-set predicted
        if len(eval_dates) > 0 and active_model and active_model.mape is not None:
            mape_label = f"Model Predicted (Test) — MAPE {active_model.mape:.1f}%"
            ax.plot(eval_dates, eval_preds, color='#fb923c', lw=1.8, linestyle='--',
                    marker='x', markersize=5, markeredgewidth=1.5,
                    label=mape_label, zorder=5, alpha=0.95)

        # vertical separator
        if training_cutoff_date is not None:
            ax.axvline(x=training_cutoff_date, color='#334155', linestyle=':', lw=1.5, alpha=0.8, zorder=2)
            ax.text(training_cutoff_date, 1.01, ' ← eval end', transform=ax.get_xaxis_transform(),
                    color='#64748b', fontsize=7, va='bottom')

        # admin new data
        if not admin_df.empty:
            if len(eval_dates) > 0:
                bx = [eval_dates[-1], admin_df['date'].iloc[0]]
                by = [eval_actual[-1], admin_df['farmgate_price'].iloc[0]]
                ax.plot(bx, by, color='#22d3ee', lw=1, linestyle=':', alpha=0.45, zorder=3)
            ax.plot(admin_df['date'], admin_df['farmgate_price'], color='#22d3ee', lw=2, linestyle='-',
                    marker='o', markersize=6, markerfacecolor='#0e7490', markeredgecolor='#22d3ee',
                    markeredgewidth=1.3, label='Admin New Data', zorder=6)

        # user forecast log
        if not log_df.empty:
            if not admin_df.empty:
                last_x, last_y = admin_df['date'].iloc[-1], admin_df['farmgate_price'].iloc[-1]
            elif len(eval_dates) > 0:
                last_x, last_y = eval_dates[-1], eval_preds[-1]
            else:
                last_x = last_y = None
            if last_x is not None:
                ax.plot([last_x, log_df['target_date'].iloc[0]],
                        [last_y, log_df['price_predicted'].iloc[0]],
                        color='#f43f5e', lw=1, linestyle=':', alpha=0.4, zorder=3)
            ax.plot(log_df['target_date'], log_df['price_predicted'], color='#f43f5e', lw=1.8, linestyle='--',
                    marker='o', markersize=4, label='User Forecast Log', zorder=7, alpha=0.9)

        # x-axis formatting
        all_unique_dates = sorted(set(all_chart_dates))
        if all_unique_dates:
            ax.set_xticks(all_unique_dates)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%Y'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=6.5, color='#94a3b8')
            pad = timedelta(days=3)
            ax.set_xlim(all_unique_dates[0] - pad, all_unique_dates[-1] + pad)
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8, color='#94a3b8')

        ax.tick_params(axis='y', colors='#94a3b8', labelsize=9)
        ax.set_ylabel('Farmgate Price (₱)', color='#94a3b8', fontsize=10)
        ax.yaxis.grid(True, color='#1e293b', linewidth=0.8, alpha=0.7)
        ax.xaxis.grid(True, color='#1e293b', linewidth=0.5, alpha=0.25)
        for s in ax.spines.values():
            s.set_visible(False)

        legend = ax.legend(facecolor='#1e293b', edgecolor='#334155', loc='upper right', fontsize=8)
        plt.setp(legend.get_texts(), color='#cbd5e1')
        plt.tight_layout(pad=1.5)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor=bg)
        graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

    # ------------------------------------------------------------------
    # 7. Active model summary
    # ------------------------------------------------------------------
    eval_summary = None
    if active_model and active_model.mape is not None:
        eval_summary = {
            'name': active_model.name,
            'order': f"({active_model.p},{active_model.d},{active_model.q})",
            'mape': round(active_model.mape, 2),
            'mae': round(active_model.mae, 2) if active_model.mae else '—',
            'rmse': round(active_model.rmse, 2) if active_model.rmse else '—',
            'aic': round(active_model.aic, 2) if active_model.aic else '—',
            'accuracy': round(100 - active_model.mape, 2),
        }

    # ------------------------------------------------------------------
    # 8. Return JSON with pure data series
    # ------------------------------------------------------------------
    response_data = {
        'no_data': False,
        'eval_summary': eval_summary,
        'eval_rows': eval_rows,                     # backward compatible
        'admin_rows': admin_rows,                   # backward compatible
        'log_rows': log_rows,                       # backward compatible
        'admin_forecast_rows': admin_forecast_rows,
        'training_cutoff_date': training_cutoff_date.strftime('%Y-%m-%d') if training_cutoff_date else None,
        'active_model_id': active_model.id if active_model else None,
        # NEW pure data series for client-side plotting
        'actual_series': actual_series,
        'test_set_predicted_series': test_set_predicted_series,
        'forecast_log_series': forecast_log_series,
    }
    if graph_base64:
        response_data['graph_base64'] = graph_base64

    return JsonResponse(response_data, safe=False)

@api_view(['POST', 'GET'])
@permission_classes([AllowAny])
def forecast_api(request):
    try:
        if request.method == 'POST':
            payload = request.data
        else:
            payload = request.query_params

        forecast_horizon = int(payload.get('forecast_horizon', 7))
        oil_price  = payload.get('oil_price_trend')
        peso_dollar = payload.get('peso_dollar_rate')

        if not oil_price or not peso_dollar:
            return Response({'success': False, 'error': 'Market data required'}, status=400)

        oil_price   = float(oil_price)
        peso_dollar = float(peso_dollar)
        oil_price_for_model = (oil_price * peso_dollar) / 1000 if oil_price > 1000 else oil_price

        active_model = TrainedModel.objects.filter(is_active=True).order_by('-training_date').first()
        if not active_model:
            return Response({'success': False, 'error': 'No active model found'}, status=404)

        arimax = ARIMAXModel()
        arimax.load_model(active_model.model_file_path)

        forecast_result = arimax.forecast(
            steps=forecast_horizon,
            use_latest_values=True,
            latest_oil=oil_price_for_model,
            latest_peso=peso_dollar,
        )

        from datetime import timedelta
        forecast_start = timezone.now().date()
        forecast_dates = [
            (forecast_start + timedelta(days=i + 1)).strftime('%Y-%m-%d')
            for i in range(forecast_horizon)
        ]

        forecast_values = forecast_result.tolist() if hasattr(forecast_result, 'tolist') else list(forecast_result)
        predicted_price = round(float(forecast_values[-1]), 2) if forecast_values else 0.0

        ForecastLog.objects.create(
            model_used=active_model,
            forecast_horizon=forecast_horizon,
            farmer_input_oil_price_trend=round(oil_price_for_model, 2),
            farmer_input_peso_dollar_rate=round(peso_dollar, 2),
            price_predicted=predicted_price,
        )

        prices_arr = np.array(forecast_values, dtype=float)
        mean_p     = float(np.mean(prices_arr))
        volatility = (float(np.std(prices_arr)) / mean_p * 100.0) if mean_p > 0 else 0.0
        start_price      = float(forecast_values[0])
        end_price        = float(forecast_values[-1])
        mean_p           = float(np.mean(prices_arr))
        volatility       = round((float(np.std(prices_arr)) / mean_p * 100.0) if mean_p > 0 else 0.0, 2)
        total_change_pct = ((end_price - start_price) / start_price * 100.0) if start_price > 0 else 0.0
        best_idx         = int(np.argmax(prices_arr))
        best_day_date    = forecast_dates[best_idx]
        best_day_price   = round(float(prices_arr[best_idx]), 2)

        if total_change_pct > 3:
            trend = 'Upward'
        elif total_change_pct < -3:
            trend = 'Downward'
        else:
            trend = 'Stable'

        if volatility > 15:
            risk_level = 'High Risk'
        elif volatility > 7:
            risk_level = 'Moderate Risk'
        else:
            risk_level = 'Low Risk'

        # Summary recommendation
        if trend == 'Upward':
            summary = (f"Prices are projected to RISE by {abs(total_change_pct):.1f}% over {forecast_horizon} days. "
                    f"Recommended to WAIT and sell closer to {best_day_date} for better returns.")
        elif trend == 'Downward':
            summary = (f"Prices are projected to DROP by {abs(total_change_pct):.1f}% over {forecast_horizon} days. "
                    f"Recommended to SELL SOON to avoid further price decline.")
        else:
            summary = (f"Prices are STABLE over the next {forecast_horizon} days. "
                    f"You have flexibility to sell based on your logistics and cash-flow needs.")

        # Sell or wait
        if trend == 'Upward':
            sell_wait = (f"Prices are trending upward. Waiting until {best_day_date} could give you "
                        f"₱{best_day_price - start_price:.2f}/kg more than selling today. "
                        f"Only wait if your copra is properly dried and stored.")
        elif trend == 'Downward':
            sell_wait = (f"Prices are trending downward. Sell as soon as possible to protect your income. "
                        f"Delaying may result in ₱{start_price - end_price:.2f}/kg loss.")
        else:
            sell_wait = "Prices are stable with minimal change expected. You can sell at your convenience."

        # Risk advisory
        if volatility > 15:
            risk_advice = (f"Price volatility is HIGH ({volatility}%). Avoid selling all your copra on a single day.")
        elif volatility > 7:
            risk_advice = (f"Moderate price fluctuations detected ({volatility}% volatility). "
                        f"Monitor weekly coconut oil price and daily peso-dollar rate changes.")
        else:
            risk_advice = f"Price forecast is stable (low volatility: {volatility}%). Conditions are favourable."
        solution = {
            'trend':       trend,
            'volatility':  volatility,
            'riskLevel':   risk_level,
            'minPrice':    round(float(np.min(prices_arr)), 2),
            'maxPrice':    round(float(np.max(prices_arr)), 2),
            'avgPrice':    round(mean_p, 2),
            'totalChange': round(total_change_pct, 2),
            'bestDay': {
                'date':  best_day_date,
                'price': best_day_price,
            },
            'summary':        summary,
            'sellWait':       sell_wait,
            'riskAdvice':     risk_advice,
            'priceRange':     (f"Over {forecast_horizon} days, prices range ₱{np.min(prices_arr):.2f}–"
                            f"₱{np.max(prices_arr):.2f}, avg ₱{mean_p:.2f}/kg. "
                            f"Use this range when negotiating with traders."),
            'marketFactors':  (f"Forecast based on Domestic Millgate Coconut Oil price ${oil_price:.2f} and and converted to (Kg) derive by peso-dollar rate ₱{peso_dollar:.2f}. "
                            f"Re-check if major market events occur."),
        }

        return Response({
            'success':         True,
            'predicted_price': predicted_price,
            'daily_forecast':  [
                {'date': d, 'predicted_price': round(float(v), 2)}
                for d, v in zip(forecast_dates, forecast_values)
            ],
            'recommendations': solution,
            'oil_price_used':  round(oil_price_for_model, 2)
        })

    except Exception as e:
        return Response({'success': False, 'error': str(e)}, status=500)
    
def recent_forecasts_api(request):
    forecasts = ForecastLog.objects.all().order_by('-created_at')[:100]
    data = []
    for f in forecasts:
        target_date = f.created_at + timedelta(days=f.forecast_horizon)
        data.append({
            'id': f.id,
            'created_at': f.created_at.isoformat(),
            'target_date': target_date.isoformat(),
            'forecast_horizon': f.forecast_horizon,
            'oil_price': float(f.farmer_input_oil_price_trend),
            'fx_rate': float(f.farmer_input_peso_dollar_rate),
            'price_predicted': float(f.price_predicted),
        })
    return JsonResponse({'count': len(data), 'forecasts': data})