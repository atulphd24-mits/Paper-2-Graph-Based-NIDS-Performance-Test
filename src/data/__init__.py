def __init__(
    self,
    dataframe,
    dataset_name="Dataset",
    label_column=None,
    attack_column=None
):

    self.df = dataframe.copy()
    self.dataset_name = dataset_name

    # Automatically detect label column
    if label_column is None:

        for col in ["label", "Label", "class", "Class"]:

            if col in self.df.columns:
                label_column = col
                break

    # Automatically detect attack column
    if attack_column is None:

        for col in ["Attack", "attack"]:

            if col in self.df.columns:
                attack_column = col
                break

    self.label_column = label_column
    self.attack_column = attack_column