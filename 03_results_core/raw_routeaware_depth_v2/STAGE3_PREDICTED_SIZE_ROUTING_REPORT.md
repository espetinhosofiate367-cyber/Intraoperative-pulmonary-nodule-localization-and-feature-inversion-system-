# Stage3 Predicted-Size Routing Evaluation

## Size Routing Quality

- Size top-1 on Stage3 test windows: `0.6894`
- Size top-2 on Stage3 test windows: `0.8258`
- Exact route match rate: `0.6894`

## Depth Performance

- GT size route balanced accuracy: `0.6066`
- Hard predicted-size route balanced accuracy: `0.5240`
- Soft predicted-size route balanced accuracy: `0.5168`
- Regression-Gaussian route (sigma=0.15) balanced accuracy: `0.5092`
- Regression-Gaussian route (sigma=0.20) balanced accuracy: `0.5104`
- Regression-Gaussian route (sigma=0.25) balanced accuracy: `0.4906`
- Top2-soft predicted-size route balanced accuracy: `0.5197`
- Temperature-0.5 soft route balanced accuracy: `0.5189`
- Temperature-0.7 soft route balanced accuracy: `0.5189`
- Temperature-0.3 soft route balanced accuracy: `0.5209`
- Hard route balanced accuracy when size route is correct: `0.6380`
- Hard route balanced accuracy when size route is wrong: `0.2383`