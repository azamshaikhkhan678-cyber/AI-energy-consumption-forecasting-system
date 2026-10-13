# main.py
"""
AI-Powered Energy Consumption Forecasting System
Author: SK Azam
Description: Tracks time-series utility loads, fits a Scikit-Learn Random Forest 
             Regression model, and draws future consumption forecasts onto a custom GUI.
"""

import sys
import random
import math
import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Bind our settings variables safely
from config.settings import *

class EnergyPredictiveMLModel:
    def __init__(self):
        records = []
        for day in range(30):
            for hour in range(24):
                base_load = 80.0 + 45.0 * math.sin(math.pi * (hour - 6) / 12)
                day_modifier = 0.85 if (day % 7 >= 5) else 1.05
                noise = random.uniform(-8.0, 8.0)
                final_mw = round((base_load * day_modifier) + noise, 2)
                records.append([hour, day % 7, day + 1, final_mw])

        columns = ['hour', 'day_of_week', 'day_of_month', 'megawatts']
        self.dataset = pd.DataFrame(records, columns=columns)

        X = self.dataset[['hour', 'day_of_week', 'day_of_month']]
        y = self.dataset['megawatts']
        self.regressor = RandomForestRegressor(n_estimators=35, random_state=42)
        self.regressor.fit(X, y)

        predictions = self.regressor.predict(X)
        self.r2_score = float(np.corrcoef(y, predictions)[0, 1] ** 2)

    def generate_24hr_forecast(self, target_day_idx):
        forecast_predictions = []
        actual_baselines = []
        for hour in range(24):
            feature_input = pd.DataFrame([[hour, target_day_idx % 7, target_day_idx]], 
                                         columns=['hour', 'day_of_week', 'day_of_month'])
            # Fixed tag index selector array mapping format to resolve scalar conversions
            pred_mw = float(self.regressor.predict(feature_input)[0])
            forecast_predictions.append(pred_mw)

            base = 80.0 + 45.0 * math.sin(math.pi * (hour - 6) / 12)
            mod = 0.85 if (target_day_idx % 7 >= 5) else 1.05
            actual_baselines.append(base * mod + random.uniform(-4.0, 4.0))
        return actual_baselines, forecast_predictions

class SmartGridForecastingDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Energy Analytics Platform - Grid Forecast Control Center")
        self.root.geometry(f"{WIDTH}x{HEIGHT}")
        self.root.configure(bg=THEME_BG_DARK)

        self.ml_engine = EnergyPredictiveMLModel()
        self.current_simulation_day = 14

        self.assemble_dashboard_layout()
        self.refresh_grid_analytics_view()

    def assemble_dashboard_layout(self):
        self.header = tk.Frame(self.root, bg=THEME_PANEL_BG, height=60)
        self.header.pack(side=tk.TOP, fill=tk.X, padx=15, pady=10)

        tk.Label(self.header, text="⚡ SMART CITIES ENERGY FORECAST CENTER", 
                 font=("Courier New", 14, "bold"), fg=COLOR_TEXT_PRIMARY, bg=THEME_PANEL_BG).pack(side=tk.LEFT, padx=20, pady=15)

        self.btn_run = tk.Button(self.header, text="🔮 RUN NEXT-DAY AI FORECAST", font=("Courier New", 9, "bold"),
                                 fg="#ffffff", bg="#0088cc", activebackground="#005588", bd=0, padx=15, command=self.step_forward_forecast_timeline)
        self.btn_run.pack(side=tk.RIGHT, padx=20, pady=12)

        self.metrics_container = tk.Frame(self.root, bg=THEME_BG_DARK)
        self.metrics_container.pack(side=tk.TOP, fill=tk.X, padx=15, pady=5)

        self.lbl_day_card = self.create_card(self.metrics_container, "TARGET SIMULATION TIMELINE", "Day 14", COLOR_TEXT_PRIMARY, 0)
        self.lbl_r2_card = self.create_card(self.metrics_container, "MODEL ACCURACY (R² RATIO)", f"{self.ml_engine.r2_score:.4f}", COLOR_ACTUAL, 1)
        self.lbl_peak_card = self.create_card(self.metrics_container, "ESTIMATED 24H PEAK SURGE", "0.0 MW", COLOR_FORECAST, 2)

        self.graph_frame = tk.Frame(self.root, bg=THEME_PANEL_BG, bd=1, relief=tk.SUNKEN)
        self.graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=15, pady=10)

        tk.Label(self.graph_frame, text="24-HOUR ENERGY CONSUMPTION MATRIX LOAD TREND LINE (ACTUAL VS ML PREDICTED FORECAST)", 
                 font=("Courier New", 8, "bold"), fg=COLOR_TEXT_SECONDARY, bg=THEME_PANEL_BG).pack(side=tk.TOP, anchor=tk.W, padx=15, pady=8)

        self.canvas = tk.Canvas(self.graph_frame, bg=THEME_BG_DARK, highlightthickness=0, bd=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

    def create_card(self, parent, title, val, color, col_idx):
        card = tk.Frame(parent, bg=THEME_PANEL_BG, bd=1, relief=tk.RAISED)
        card.grid(row=0, column=col_idx, padx=8, pady=5, sticky="nsew")
        parent.grid_columnconfigure(col_idx, weight=1)
        tk.Label(card, text=title, font=("Courier New", 8, "bold"), fg=COLOR_TEXT_SECONDARY, bg=THEME_PANEL_BG).pack(pady=6)
        lbl = tk.Label(card, text=val, font=("Courier New", 18, "bold"), fg=color, bg=THEME_PANEL_BG)
        lbl.pack(pady=2)
        return lbl

    def step_forward_forecast_timeline(self):
        self.current_simulation_day = (self.current_simulation_day + 1) if (self.current_simulation_day < 30) else 1
        self.lbl_day_card.config(text=f"Day {self.current_simulation_day}")
        self.refresh_grid_analytics_view()

    def refresh_grid_analytics_view(self):
        actual_data, forecast_data = self.ml_engine.generate_24hr_forecast(self.current_simulation_day)
        peak_value = max(forecast_data)
        self.lbl_peak_card.config(text=f"{peak_value:.2f} MW")
        self.canvas.delete("all")
        
        c_width = self.canvas.winfo_width() if self.canvas.winfo_width() > 100 else 900
        c_height = self.canvas.winfo_height() if self.canvas.winfo_height() > 100 else 400
        pad_x, pad_y = 50, 40
        plot_w = c_width - (pad_x * 2)
        plot_h = c_height - (pad_y * 2)

        self.canvas.create_line(pad_x, pad_y, pad_x, c_height - pad_y, fill=COLOR_TEXT_SECONDARY, width=1)
        self.canvas.create_line(pad_x, c_height - pad_y, c_width - pad_x, c_height - pad_y, fill=COLOR_TEXT_SECONDARY, width=1)

        for i in range(5):
            y_val = 40 + i * 30
            y_pos = c_height - pad_y - (y_val / 160.0) * plot_h
            self.canvas.create_text(pad_x - 20, y_pos, text=f"{y_val}M", fill=COLOR_TEXT_SECONDARY, font=("Courier New", 7))
            self.canvas.create_line(pad_x - 4, y_pos, pad_x, y_pos, fill=COLOR_TEXT_SECONDARY)

        for hour in range(24):
            x_pos = pad_x + (hour / 23.0) * plot_w
            if hour % 4 == 0:
                self.canvas.create_text(x_pos, c_height - pad_y + 15, text=f"{hour:02d}:00", fill=COLOR_TEXT_SECONDARY, font=("Courier New", 7))
                self.canvas.create_line(x_pos, c_height - pad_y, x_pos, c_height - pad_y + 4, fill=COLOR_TEXT_SECONDARY)

        actual_points = []
        forecast_points = []
        for hour in range(24):
            x_pos = pad_x + (hour / 23.0) * plot_w
            y_act_pos = c_height - pad_y - (actual_data[hour] / 160.0) * plot_h
            y_fore_pos = c_height - pad_y - (forecast_data[hour] / 160.0) * plot_h
            actual_points.extend([x_pos, y_act_pos])
            forecast_points.extend([x_pos, y_fore_pos])

        self.canvas.create_line(actual_points, fill=COLOR_ACTUAL, width=2, dash=(2, 2))
        self.canvas.create_line(forecast_points, fill=COLOR_FORECAST, width=3)

        self.canvas.create_line(60, 20, 90, 20, fill=COLOR_ACTUAL, width=2, dash=(2,2))
        self.canvas.create_text(150, 20, text="Historical Load (Actual)", fill=COLOR_TEXT_SECONDARY, font=("Courier New", 8), anchor=tk.W)
        self.canvas.create_line(300, 20, 330, 20, fill=COLOR_FORECAST, width=3)
        self.canvas.create_text(390, 20, text="Scikit-Learn Forecast (AI)", fill=COLOR_TEXT_SECONDARY, font=("Courier New", 8), anchor=tk.W)

        if peak_value > PEAK_THRESHOLD_MW:
            messagebox.showwarning("GRID OPERATION ALARM", 
                                   f"CRITICAL OVERLOAD RISK ALERT!\n\nForecasted Peak Load: {peak_value:.2f} MW\nSafety Limit: {PEAK_THRESHOLD_MW} MW\n\nAI recommends engaging auxiliary renewable grid reserves instantly to balance peak city demand.")

if __name__ == "__main__":
    grid_window = tk.Tk()
    forecaster_app = SmartGridForecastingDashboard(grid_window)
    grid_window.after(200, forecaster_app.refresh_grid_analytics_view)
    grid_window.mainloop()
