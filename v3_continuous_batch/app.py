from common.server import create_app
from v3_continuous_batch.scheduler import ContinuousBatchScheduler

app = create_app("v3-continuous-batch", ContinuousBatchScheduler)
