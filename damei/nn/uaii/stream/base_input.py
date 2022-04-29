from .base_queue import AbstractQue


class AbstractInput(object):
    name = 'default_input_name'
    status = 'stopped'
    description = 'default_input_description'

    def __init__(self, *args, **kwargs):
        que = kwargs.pop('que', None)
        if que:
            maxlen = que.get('maxlen', 5)
            self.que = AbstractQue(maxlen, *args, **kwargs)
        else:
            self.que = None
