import time

class Timer:
    """
    Utility to measure elapsed execution time in milliseconds.
    """
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.interval = (self.end - self.start) * 1000.0 # to ms
