import numpy as np
from time import time

class PrCol:
    def __init__(self):
        self.black = "\033[30m"
        self.red = "\033[91m"
        self.green = "\033[92m"
        self.yellow = "\033[93m"
        self.blue = "\033[94m"
        self.magenta = "\033[95m"
        self.cyan = "\033[96m"
        self.white = "\033[97m"
        self.bright_black = "\033[90m"
        self.bright_red = "\033[91;1m"
        self.bright_green = "\033[92;1m"
        self.bright_yellow = "\033[93;1m"
        self.bright_blue = "\033[94;1m"
        self.bright_magenta = "\033[95;1m"
        self.bright_cyan = "\033[96;1m"
        self.bright_white = "\033[97;1m"
        self.end = "\033[0m"

Color = PrCol()

class CircularBuffer:
    def __init__(self, dimension):
        self.dimension = dimension
        self.buffer = []

    def add_element(self, val):
        if len(self.buffer) < self.dimension:
            self.buffer.append(val)
        else:
            self.buffer = self.buffer[1:] + [val]
    
    def get_mean(self):
        if len(self.buffer) == 0:
            return np.nan
        return np.mean(self.buffer)
    

class LoadBar():
    t1 = np.inf

    def __init__(self) -> None:
        self.timeBuffer = CircularBuffer(50)

    def _getLoadBar(self, actual, total, startchar="|", loadchar=f"{Color.green}█{Color.end}", uncompletedchar=f"{Color.black}█{Color.end}", endchar="|", length=20):
        percent_completed = int(actual / total * 100)
        completed_length = int(length * actual / total)
        uncompleted_length = length - completed_length
        load_bar = (
            startchar +
            loadchar * completed_length +
            uncompletedchar * uncompleted_length +
            endchar
        )
        return load_bar

    def loadBarIJ(self, i, end_i, j, end_j, prefix = "", suffix = ""):
        eta = self.timeBuffer.get_mean() * (end_i - i)
        return f'{prefix} {self._getLoadBar(i + 1,end_i)}{self._getLoadBar(j + 1, end_j, startchar="",loadchar=f"{Color.yellow}█{Color.end}",length=7)} ' + \
                            f'({i + 1}/{end_i}) ({self.timeBuffer.get_mean():.2f}s avg time) ' + \
                            f'ETA: {seconds_to_hhmmss(eta)} - {suffix}'
    
    def loadBarI(self, i, end_i, prefix = "", suffix = ""):
        eta = self.timeBuffer.get_mean() * (end_i - i)
        return f'{prefix} {self._getLoadBar(i + 1,end_i)}' + \
                            f'({i + 1}/{end_i}) ({self.timeBuffer.get_mean():.2f}s avg time) ' + \
                            f'ETA: {seconds_to_hhmmss(eta)} - {suffix}'

    def tick(self):
        self.t1 = time()

    def tock(self):
        self.timeBuffer.add_element(time() - self.t1)

def darken(hex_color, factor=0.5):
    # Ensure the factor is within the valid range [0, 1]
    factor = max(0, min(1, factor))

    # Convert hex to RGB
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)

    # Darken each RGB component
    r = int(r * (1 - factor))
    g = int(g * (1 - factor))
    b = int(b * (1 - factor))

    # Convert back to hex and return
    darkened_hex = "#{:02X}{:02X}{:02X}".format(r, g, b)
    return darkened_hex

def seconds_to_hhmmss(secs, minFormat=True):
    try:
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = int(secs % 60)
        mm = m + 60 * h
        ss = s + 60 * mm
        if minFormat: 
            if mm==0: return "~{:02d} sec".format((ss // 10 + 1) * 10)
            else: return "~{:02d} min".format(mm+1)
        else: return "{:02d}:{:02d}:{:02d}".format(h, m, s)
    except ValueError:
        return np.nan