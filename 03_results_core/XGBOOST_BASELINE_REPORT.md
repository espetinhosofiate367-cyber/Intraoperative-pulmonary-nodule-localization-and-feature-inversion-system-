# XGBoost Baseline Report

## Protocol
- Input window: `10 x 1 x 12 x 8`
- Stride: `2`
- Split: `1.CSV + 2.CSV` development, `3.CSV` final test
- Detection label: center frame inside positive segment

## Detection
- XGBoost test AUC: `0.8199`
- XGBoost test AP: `0.5063`
- XGBoost test F1@val threshold: `0.6185`
- Dummy prior test AUC: `0.5000`
- Dummy prior test AP: `0.2565`

## Size / Depth (GT-positive test windows)
- XGBoost size top1: `0.6701`
- XGBoost size top2: `0.8023`
- XGBoost size MAE: `0.1472 cm`
- XGBoost depth coarse acc: `0.5151`
- Dummy size top1: `0.2102`
- Dummy size MAE: `0.4515 cm`
- Dummy depth acc: `0.3434`

## Notes
- This baseline uses only handcrafted window features plus XGBoost.
- It is intended to anchor the later neural-network comparison.