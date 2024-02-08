# ************************************************************************************************************************* #
#                                           Copyright (C) 2022 Jorge Brenes Alfaro.
#                                           EL5617 Trabajo Final de Graduación.
#                                           Escuela de Ingeniería Electrónica.
#                                           Tecnológico de Costa Rica.
# *************************************************************************************************************************** #

#   This file contains the timer class retreieve from [1]. It purpose is to call a function a number
#   of times for a given time.

#   Libraries
import time
import threading

# ********************************** Timer Class **********************************#
class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.next_call = time.time()
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self.next_call += self.interval
            self._timer = threading.Timer(self.next_call - time.time(), self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

# ********************************** REference **********************************#
#   [1] Erauol, "What is the best way to repeatedly execute a function every x seconds?, 2016.
#   [Online]. Retrieve from: https://stackoverflow.com/questions/474528/what-is-the-best
#   -way-to-repeatedly-execute-a-function-every-x-seconds
