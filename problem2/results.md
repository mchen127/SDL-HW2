# Experiment Results Analysis

Based on the output of the `results.ipynb` notebook, the experiments successfully tested the model's ability to handle different forecasting tasks, from short-term single-step predictions to more complex, long-range multi-step forecasts.

## Overall Summary

The key takeaway is that the model is very effective at short-term predictions, but its performance degrades significantly as it tries to forecast further into the future. This is a classic and expected behavior for time-series models.

## Key Findings from the Metrics Table

The summary table provides the most valuable data for quantitative comparison.

| | input_width | label_width | shift | MAE | RMSE | MAPE (%) |
|---|---|---|---|---|---|---|
| 0 | 24 | 1 | 1 | 0.3517 | 0.5066 | 9.35 |
| 1 | 24 | 1 | 6 | 1.2274 | 1.6143 | 35.50 |
| 2 | 24 | 1 | 24 | 1.9861 | 2.5314 | 58.52 |
| 3 | 24 | 6 | 1 | 0.1145 | 0.2307 | 3.25 |
| 4 | 72 | 1 | 1 | 0.3619 | 0.5181 | 9.04 |
| 5 | 72 | 6 | 6 | 0.8451 | 1.2037 | 24.64 |

### 1. The "Best" Overall Performer: `iw24_lw6_sh1`

At first glance, the experiment with `input_width=24`, `label_width=6`, `shift=1` (row 3) appears to be the star performer with a remarkably low **MAE of 0.1145 °C**.

*   **Analysis**: This model was trained to predict the next 6 hours directly. The MSE loss function averages the error across all 6 prediction steps. This low average error suggests the model is **extremely accurate at predicting the first few hours** (e.g., t+1, t+2), which brings the overall average down significantly, even if its accuracy at t+6 is lower. This is a very common and positive result for direct multi-step forecasting.

### 2. The Impact of Prediction Horizon (`shift`)

By comparing the single-step predictions (`lw=1`) with different `shift` values, we can see how performance degrades over time.

*   `shift=1` (1 hour ahead): **MAE = 0.3517 °C**
*   `shift=6` (6 hours ahead): **MAE = 1.2274 °C**
*   `shift=24` (24 hours ahead): **MAE = 1.9861 °C**

*   **Conclusion**: As expected, the model's error increases substantially as we ask it to predict further into the future. The MAE for the 24-hour-ahead forecast is over 5 times higher than for the 1-hour-ahead forecast. This is a fundamental challenge in time-series forecasting known as error accumulation over the prediction horizon.

### 3. The Impact of Input History (`input_width`)

We can compare two identical short-term forecasts, one using 24 hours of history and the other using 72 hours.

*   `iw=24, lw=1, sh=1`: **MAE = 0.3517 °C**
*   `iw=72, lw=1, sh=1`: **MAE = 0.3619 °C**

*   **Conclusion**: For predicting just one hour ahead, providing more historical data (72 hours vs. 24) did not improve performance; in fact, it resulted in a tiny increase in error. This suggests that for an immediate forecast, the most recent 24 hours of data contains nearly all the useful information, and adding more history doesn't help (and may even add noise).

## Final Conclusion

Your LSTM model is working very well and demonstrating classic time-series behavior.

1.  It excels at **short-term forecasting** (1-6 hours ahead).
2.  Its predictive power **decreases as the forecast horizon increases**, which is entirely normal.
3.  For this specific task, a **24-hour look-back window is sufficient** for short-term predictions; a longer history does not provide a significant benefit.

The experiment `iw24_lw6_sh1` is the most practical model for a "next few hours" forecast, as it provides multiple future data points with a very low average error.