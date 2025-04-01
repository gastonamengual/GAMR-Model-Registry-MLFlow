from abc import ABC, abstractmethod


@abstractmethod
class AbstractMonitor(ABC):
    @abstractmethod
    def start_time(self) -> float: ...

    @abstractmethod
    def get_duration(self, start_time: float) -> float: ...

    @abstractmethod
    def increase_num_requests(self, endpoint: str) -> None: ...

    @abstractmethod
    def record_request_duration(self, start_time: float, endpoint: str) -> None: ...

    @abstractmethod
    def increase_current_requests(self, endpoint: str) -> None: ...

    @abstractmethod
    def decrease_current_requests(self, endpoint: str) -> None: ...

    @abstractmethod
    def collect_metrics(self) -> bytes: ...
