"""Point-in-time data splitting helpers."""


def purged_temporal_split(
    data,
    train_start,
    validation_start,
    validation_end,
):
    """Return training and validation rows without label-window overlap."""
    candidate_train = data["ReferenceDate"].between(
        train_start, validation_start, inclusive="left"
    )
    train_mask = candidate_train & data["LabelEndDate"].le(validation_start)
    validation_mask = data["ReferenceDate"].between(
        validation_start, validation_end
    )
    return data.loc[train_mask].copy(), data.loc[validation_mask].copy()

