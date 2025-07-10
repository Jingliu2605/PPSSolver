import time
from datetime import timedelta

from pymoo.model.termination import Termination


class TimeBasedTermination(Termination):

    def __init__(self, seconds=0, milliseconds=0, minutes=0, hours=0) -> None:
        super().__init__()
        self.start = None
        self.now = None

        max_time = timedelta(0, seconds, 0, milliseconds, minutes, hours, 0)
        self.max_time = max_time.total_seconds()

        #if isinstance(max_time, str):
        #    self.max_time = time_to_int(max_time)
        #elif isinstance(max_time, int) or isinstance(max_time, float):
        #    self.max_time = max_time
        #else:
        #    raise Exception("Either provide the time as a string or an integer.")

    def do_continue(self, algorithm):
        if self.start is None:
            self.start = time.time() # process_time()
        self.now = time.time()
        return self.now - self.start < self.max_time