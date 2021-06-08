"""
Class that will execute commands when a file change is detected.
"""
import threading

from watchdog.events import FileSystemEventHandler, FileSystemEvent


class PlotTrigger(FileSystemEventHandler):
    def __init__(self):
        self.lock = threading.Lock()
        self.update = False
        self.filename = None

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory:
            with self.lock:
                self.update = True
                self.filename = event.src_path
