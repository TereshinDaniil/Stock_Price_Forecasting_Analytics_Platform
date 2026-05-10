import numpy as np
import torch
from chronos import ChronosPipeline


def chronos_forecast(
    series,
    horizon,
    model_name="amazon/chronos-t5-small",
    num_samples=20,
    device=None,
    random_state=42,
):
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32,
    )

    context = torch.tensor(np.array(series), dtype=torch.float32)
    forecast = pipeline.predict(
        context,
        horizon,
        num_samples=num_samples,
    )

    forecast = forecast[0].cpu().numpy()
    preds = forecast.mean(axis=0)
    return preds, pipeline
