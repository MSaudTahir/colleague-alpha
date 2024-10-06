from discord.sinks.core import Filters, Sink, default_filters
import numpy as np

class StreamSink(Sink):
    def __init__(self, *, filters=None):
        if filters is None:
            filters = default_filters
        self.filters = filters
        Filters.__init__(self, **self.filters)
        self.vc = None
        self.audio_data = {}
        self.audio_buffer = []

    def write(self, data, user):
        if user not in self.audio_data:
            self.audio_data[user] = []
        if data:
            self.audio_data[user].append(data)
            self.audio_buffer.append(data) 
        else:
            self.audio_buffer.append(np.zeros((960*2,), dtype=np.int16).tobytes())

    def clear_audio_data(self):
        self.audio_data = {}
        self.audio_buffer = []

    def cleanup(self):
        self.finished = True

    def get_all_audio(self):
        pass

    def get_user_audio(self, user):
        pass