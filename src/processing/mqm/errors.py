"""Distinguish configuration/connection failures from errors of one fragment."""
class MQMRequestError(RuntimeError):
    def __init__(self, message, *, fatal=True):
        super().__init__(message)
        self.fatal = fatal
