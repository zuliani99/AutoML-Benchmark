def get_target(train, test):
    for c in train.columns:
        if c not in test.columns:
            return c