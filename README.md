# Personal Project

## Overview
**Language:** Python  
**Tech Stack:** SQLite, NumPy, Pandas, Matplotlib  
**API Used:** Yahoo Finance (`yfinance`)  

This project uses **Pandas** and **SQLite** to maintain a local database for long-term financial data analysis, while fetching real-time data from the public **Yahoo Finance API** on demand. It provides **2D and 3D visualizations** to analyze trends across daily, hourly, and monthly intervals.

## Features
- **Exponential Moving Average (EMA)** for trend smoothing:
  - β = 0.3 → minute-level  
  - β = 0.24 → daily  
  - β = 0.18 → weekly  
  - β = 0.15 → biweekly  
  - β = 0.13 → monthly  
- **Deviation Calculation:**
  - deviation_new = (1 - α) * deviation_current + α * |difference|
  - where α ≈ 0.3
- **Trend Analysis Granularity:**
- 5-minute EMA for hourly trends  
- Hourly EMA for daily trends  
- Daily EMA for weekly/monthly trends  
- **Data Interpolation:**  
- Linear interpolation for hourly/daily trends  
- Cubic spline interpolation for fine adjustments  

## Dependencies
- Python 3.13
- NumPy  
- Pandas  
- Matplotlib  
- SQLite3 (built-in)  
- Yahoo Finance API (`yfinance`)  


## usage
