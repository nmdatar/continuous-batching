from common.server import create_app
from v2_dynamic_batch.scheduler import DynamicBatchScheduler

app = create_app("v2-dynamic-batch", DynamicBatchScheduler)
