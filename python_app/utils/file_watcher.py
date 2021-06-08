"""
Class that will execute commands when a file change is detected.
"""

from watchdog.events import FileSystemEventHandler, FileSystemEvent


class PlotTrigger(FileSystemEventHandler):
    def __init__(self):
        self.update = False
        filename = ""

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory:
            self.update = True
            self.filename = event.src_path
            print(event.src_path)
            print("Updated parameters")
