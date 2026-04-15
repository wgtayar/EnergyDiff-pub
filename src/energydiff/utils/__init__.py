from datetime import datetime

__all__ = ['io', 'plot', 'eval', 'generic','configuration', 'sample', 'argument_parser']

def generate_time_id():
    return datetime.now().strftime("%Y%m%d-%H%M")
