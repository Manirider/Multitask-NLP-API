import time

from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

API_REQUESTS_TOTAL = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "status"],
)

API_REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "API request latency in seconds", [
        "endpoint"]
)


class MetricsMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request, call_next):
        start_time = time.time()

        response = await call_next(request)

        latency = time.time() - start_time

        endpoint = request.url.path
        method = request.method
        status = str(response.status_code)

        API_REQUESTS_TOTAL.labels(
            endpoint=endpoint, method=method, status=status).inc()
        API_REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

        return response
