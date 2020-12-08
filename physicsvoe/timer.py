import time

class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *_):
        duration = time.time() - self.start
        mins = int(duration // 60)
        secs = duration - (mins*60)
        print(f'Timer {self.name} finished   {mins}:{secs:.3f}')
