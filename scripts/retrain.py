from mlops_air_quality_forecast.retraining import retrain

result = retrain()
for key, value in result.items():
    print(f"{key}: {value}")
