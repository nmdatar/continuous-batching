from common.server import create_app
from v1_serial.scheduler import SerialScheduler

app = create_app("v1-serial", SerialScheduler)
