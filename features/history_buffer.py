from collections import deque

class PredictionBuffer:
    def __init__(self, maxlen=10):
        self.buffer = deque(maxlen=maxlen)

    def add(self, prediction):
        self.buffer.append(prediction)

    def majority_vote(self):
        if not self.buffer:
            return None
        return max(set(self.buffer), key=self.buffer.count)
