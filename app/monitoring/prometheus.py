import time
from dataclasses import dataclass, field

from prometheus_client import Counter, Gauge, Histogram, generate_latest

from .abstract import AbstractMonitor


@dataclass
class PrometheusMonitor(AbstractMonitor):
    num_requests: Counter = field(init=False)
    request_duration_seconds: Histogram = field(init=False)
    current_requests: Gauge = field(init=False)

    def __post_init__(self) -> None:
        self.num_requests = Counter(
            "flower_model_service_num_requests",
            "Total number of requests",
            labelnames=["endpoint"],
        )
        self.request_duration_seconds = Histogram(
            "flower_model_service_request_duration_seconds",
            "Histogram of request duration in seconds",
            labelnames=["endpoint"],
        )
        self.current_requests = Gauge(
            "flower_model_service_current_requests",
            "Current number of requests",
            labelnames=["endpoint"],
        )

    def start_time(self) -> float:
        return time.time()

    def get_duration(self, start_time: float) -> float:
        return time.time() - start_time

    def increase_num_requests(self, endpoint: str) -> None:
        self.num_requests.labels(endpoint=endpoint).inc()

    def record_request_duration(self, start_time: float, endpoint: str) -> None:
        duration = self.get_duration(start_time)
        self.request_duration_seconds.labels(endpoint=endpoint).observe(duration)

    def increase_current_requests(self, endpoint: str) -> None:
        self.current_requests.labels(endpoint=endpoint).inc()

    def decrease_current_requests(self, endpoint: str) -> None:
        self.current_requests.labels(endpoint=endpoint).dec()

    def collect_metrics(self) -> bytes:
        return generate_latest()  # type: ignore


PROMETHEUS_MONITOR = PrometheusMonitor()
