"""Simulated door lifecycle; no physical lock or GPIO is controlled."""


class DoorCycle:
    def __init__(self, mode="continuous", open_seconds=3.0, closed_seconds=1.5):
        self.mode = mode
        self.open_seconds = open_seconds
        self.closed_seconds = closed_seconds
        self.reset()

    def reset(self):
        self.phase = "waiting"
        self.deadline = None
        self.completed = False

    def start(self, now):
        if self.phase != "waiting" or self.completed:
            return False
        self.phase = "open"
        self.deadline = now + self.open_seconds
        return True

    def advance(self, now):
        if self.deadline is None or now < self.deadline:
            return None
        if self.phase == "open":
            self.phase = "closed"
            self.deadline = now + self.closed_seconds
            return "closed"
        if self.phase == "closed":
            self.phase = "waiting"
            self.deadline = None
            self.completed = self.mode == "once"
            return "finished" if self.completed else "reset"
        return None
