# MIT License

# Copyright (c) 2024 Intelligent Robot Motion Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Simple timer from https://github.com/jannerm/diffuser/blob/main/diffuser/utils/timer.py
"""

import time
class Timer:
    def __init__(self):
        self._start = time.time()

    def __call__(self, reset=True):
        now = time.time()
        diff = now - self._start
        if reset:
            self._start = now
        return diff
def sec2HMS(seconds):
    """ 
    translate duration in seconds to hour, minute, second format.
    """
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    hours = int(hours)
    minutes = int(minutes)
    seconds=int(seconds)
    if hours > 0:
        return f"{int(hours):02d}h:{int(minutes):02d}m:{seconds:02d}s"
    else:
        return f"{int(minutes):02d}m:{seconds:02d}s"


# appended by ReinFlow Authors
def current_time():
    from datetime import datetime
    # Get current time
    now = datetime.now()
    # Format the time to the desired pattern
    formatted_time = now.strftime("%y-%m-%d-%H-%M-%S")
    return formatted_time