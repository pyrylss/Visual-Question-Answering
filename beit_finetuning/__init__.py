def get_runner(__C):
    if __C.RUN_MODE == 'finetune':
        from .finetune import Runner
    elif __C.RUN_MODE == 'heuristics':
        from .heuristics import Runner
    else:
        raise NotImplementedError
    runner = Runner(__C)
    return runner