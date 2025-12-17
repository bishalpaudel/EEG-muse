import os
from pylsl import resolve_byprop

def resolve_stream(timeout=5):
    target = os.environ.get('MUSE_STREAM_NAME')
    if target:
        print(f"Resolving stream by name: {target}")
        return resolve_byprop('name', target, timeout=timeout)
    else:
        print("Resolving stream by type: EEG")
        return resolve_byprop('type', 'EEG', timeout=timeout)
