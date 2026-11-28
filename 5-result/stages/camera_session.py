"""Own one camera flow at a time, with explicit entry/exit for enrollment."""

from enrollment import EnrollmentPipeline
from pipeline import FacePipeline


class CameraSession:
    def __init__(self, engine, purpose='authenticate', session_id=0, enrollment_name=None, **kwargs):
        self.engine, self.options = engine, kwargs
        self.active = self._create(purpose, session_id, enrollment_name)

    def _create(self, purpose, session_id, name=None):
        options = dict(self.options)
        if purpose == 'enroll':
            options.pop('save_interval', None)
            return EnrollmentPipeline(self.engine, name, session_id=session_id, **options)
        pipeline = FacePipeline(self.engine, purpose=purpose, **options)
        pipeline.session_id = session_id
        return pipeline

    @property
    def session_id(self):
        return self.active.session_id

    @property
    def profile(self):
        return self.active.profile

    @property
    def threshold(self):
        return self.active.threshold

    @threshold.setter
    def threshold(self, value):
        if self.active.purpose != 'enroll':
            self.active.threshold = value
            self.options['threshold'] = value

    @property
    def save_interval(self):
        return self.options.get('save_interval', 1.)

    @save_interval.setter
    def save_interval(self, value):
        self.options['save_interval'] = value

    def set_purpose(self, purpose, session_id=None, enrollment_name=None, cancel_enrollment=False):
        if self.active.purpose == 'enroll' and not cancel_enrollment:
            return False  # includes terminal errors and successful DB commits awaiting confirmation
        replacement = self._create(purpose, self.session_id if session_id is None else session_id,
                                   enrollment_name)
        self.active.close()
        self.active = replacement
        return True

    def restart_enrollment(self, session_id):
        return self.active.purpose == 'enroll' and self.active.restart(session_id)

    def begin_auth_attempt(self, diagnostic=False):
        if self.active.purpose == 'authenticate':
            self.active.begin_auth_attempt(diagnostic)
            return self.active.front_threshold

    def finish_auth_attempt(self, outcome, reason):
        if self.active.purpose == 'authenticate':
            self.active.finish_auth_attempt(outcome, reason)

    def request_manual_capture(self, session_id, direction, count, requested_at):
        return (self.active.purpose == 'enroll'
                and self.active.request_manual_capture(session_id, direction, count, requested_at))

    @property
    def manual_capture_status(self):
        return self.active.manual_status if self.active.purpose == 'enroll' else '얼굴 등록 중에만 수동 촬영할 수 있습니다.'

    def reset_standby(self):
        return self.active.reset_standby()

    def process(self, frame):
        return self.active.process(frame)

    def close(self):
        self.active.close()
