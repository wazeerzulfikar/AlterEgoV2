def flatten(sequence_groups):
    return np.concatenate(sequence_groups, axis=0)

def combine(datasets):
    assert len(set(map(len, datasets))) == 1
    return [reduce(lambda a,b: list(a)+list(b),
                   map(lambda sg: sg[i], datasets)) for i in range(len(datasets[0]))]

def join(datasets):
    return reduce(lambda a,b: list(a)+list(b), datasets)