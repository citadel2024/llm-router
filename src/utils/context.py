import json
import uuid
import contextvars
from typing import Optional
from datetime import datetime
from dataclasses import field, asdict, dataclass

router_context = contextvars.ContextVar("Router context")


@dataclass
class RouterContext:
    """
    We should avoid using router context as much as possible to prevent hiding too many implementation details,
    which can make the code difficult to read. It should only be used to store data that aids in link processing and analysis.
    """

    # The request_id is automatically generated during initialization and should not be passed as a parameter to the constructor.
    request_id: str = field(init=False)
    # Before invoke a provider, we create a context with start_time, so that we can manage the usage(RPM/TPM) at the current minute.
    model_group: str
    token_count: int
    start_time: datetime = field(init=False)
    provider_id: Optional[str] = None

    def __post_init__(self):
        self.request_id = str(uuid.uuid4())
        self.start_time = datetime.now()

    def start_minute_str(self):
        return self.start_time.strftime("%Y%m%d%H%M")

    def serialize(self):
        data = asdict(self)
        data["start_time"] = self.start_time.isoformat()
        return json.dumps(data)

    def update_start_time(self):
        self.start_time = datetime.now()

    def update_model_group(self, group: str):
        self.model_group = group

    def update_provider_id(self, provider_id: str):
        self.provider_id = provider_id
