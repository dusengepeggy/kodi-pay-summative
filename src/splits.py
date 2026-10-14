from sklearn.model_selection import train_test_split

SEED = 42


def make_splits(data, test_size=0.15, val_size=0.15, seed=SEED):
    """Stratified train/validation/test split on PayTier and Year together."""
    strata = data["Year"].astype(str) + "_" + data["PayTier"].astype(str)

    train_val, test = train_test_split(
        data, test_size=test_size, stratify=strata, random_state=seed
    )

    val_fraction = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_fraction,
        stratify=strata.loc[train_val.index], random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)