class DictSafe(dict):

    def __init__(self, data={}):
        dict.__init__(self, data)
        for key, value in data.items():
            if isinstance(value, dict):
                self[key] = DictSafe(value)

    def __getitem__(self, key):
        return self.get(key, [])

TASK_TO_SPLIT = {
    'ok': {
        'finetune': {
            'train_split': ['oktrain'],
            'eval_split': ['oktest'],
        }
    },
    'aok_val': {
        'finetune': {
            'train_split': ['aoktrain'],
            'eval_split': ['aokval'],
        }
    },
    'aok_test': {
        'finetune': {
            'train_split': ['aoktrain'],
            'eval_split': ['aoktest'],
        }
    },
}
TASK_TO_SPLIT = DictSafe(TASK_TO_SPLIT)

SPLIT_TO_IMGS = {
    'oktrain': 'train2014',
    'oktest': 'val2014',
    'aoktrain': 'train2017',
    'aokval': 'val2017',
    'aoktest': 'test2017',
}


if __name__ == '__main__':
    print(TASK_TO_SPLIT['okvqa']['test']['train_split'])