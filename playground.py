import numpy as np
from matplotlib import pyplot as plt
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer
import time
import globby

SP_POINTS = 200

plt.ion()
fig, ax = plt.subplots(1)
plot, = ax.plot([0]*SP_POINTS)
plt.draw()  # non-blocking drawing
plt.pause(.001)  # This line is essential, without it the plot won't be shown

event_handler = FileSystemEventHandler()
def my_dispatch(event: FileSystemEvent):
    if not event.is_directory:
        print(".")
        globby.update=True
        globby.filename=event.src_path
event_handler.on_modified = my_dispatch
observer = Observer()
observer.schedule(event_handler, "./csrc/dump")
observer.start()

try:
    while True:
        time.sleep(1)
        if globby.update:
            globby.update = False
            arr = np.transpose(np.loadtxt(globby.filename, skiprows=1))
            plot.set_ydata(arr[1])
            ax.relim()
            ax.autoscale_view()
            ax.set_title(globby.filename)
            plt.draw()  # non-blocking drawing
            plt.pause(.001)  # This line is essential, without it the plot won't be shown
            
except KeyboardInterrupt:
    observer.stop()
observer.join()

