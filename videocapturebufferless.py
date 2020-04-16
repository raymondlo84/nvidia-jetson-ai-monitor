import cv2
import queue
import threading
import time

# bufferless VideoCapture based on OpenCV
class VideoCaptureBufferless:
  thread_id = 0
  stop_capture = 0
  counter = 0
  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    #t.daemon = True
    t.start()
    self.thread_id = t

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      #self.counter = self.counter + 1
      #print(self.counter) 
      #use this to see your frame rate
      if self.stop_capture == 1:
        break
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

  def release(self):
    self.stop_capture=1
    self.thread_id.join()
    self.cap.release()

