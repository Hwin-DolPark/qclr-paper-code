import numpy as np


def random_sequence_augmentation(x_i, y_true, seq_length, min_length=3,
                                 multiplier_0=2, multiplier_1=3, seed=42):
    """
        Sequence Augmentation Module - Random Sequence Augmentation
        x_i (np.array): input data (Batch, sequence, feature)
        y_ture (np.array): output label (mortality)
        seq-Length (np.array, int): Actual time series length for each patient (length)
        x_id:
        min_length: Minimum sequence length
        multiplier_0: Augmentation factor for class 0 (survivors)
        multiplier_1: Augmentation factor for class 1 (deceased patients)
        """
    y_true = y_true.squeeze()
    batch_size, max_seq_length, feature_dim = x_i.shape

    # Finding indexes by class (mortality)
    class_0_indices = np.where(y_true == 0)[0]
    class_1_indices = np.where(y_true == 1)[0]

    # Set the number of augmented data
    num_aug_class_0 = int(len(class_0_indices) * (multiplier_0 - 1))
    num_aug_class_1 = int(len(class_1_indices) * (multiplier_1 - 1))

    # Performing per-class augmentation
    x_i_class_0 = x_i[class_0_indices]
    x_i_class_1 = x_i[class_1_indices]
    seq_length_class_0 = seq_length[class_0_indices]
    seq_length_class_1 = seq_length[class_1_indices]

    # Combine original and augmented data
    augmented_sequences_0, random_lengths_0 = augment_class(x_i_class_0,
                                                            seq_length_class_0,
                                                            num_aug_class_0,
                                                            min_length, seed)
    augmented_sequences_1, random_lengths_1 = augment_class(x_i_class_1,
                                                            seq_length_class_1,
                                                            num_aug_class_1,
                                                            min_length, seed)

    # Combine original and augmented data
    augmented_x_i = np.concatenate(
        [x_i, augmented_sequences_0, augmented_sequences_1], axis=0)
    augmented_y_true = np.concatenate(
        [y_true, np.zeros(num_aug_class_0, dtype=y_true.dtype),
         np.ones(num_aug_class_1, dtype=y_true.dtype)], axis=0)

    return augmented_x_i, augmented_y_true


def augment_class(x_i_class, seq_length_class, num_aug, min_length, seed):
    """
    Augment time series data for a class (mortality).
    """
    batch_size_class = x_i_class.shape[0]
    feature_dim = x_i_class.shape[2]

    # An array to store the sequence to be augmented
    augmented_sequences = np.zeros((num_aug, x_i_class.shape[1], feature_dim))
    random_lengths = np.zeros(num_aug, dtype=np.int32)

    for i in range(num_aug):
        # Select a sample at random
        batch_index = np.random.randint(0, batch_size_class)
        actual_length = seq_length_class[batch_index]

        if actual_length < min_length:
            # If sequence length is less than min_length - decrease length by one
            start_value = x_i_class[batch_index, 0:1, :]
            end_value = x_i_class[batch_index, actual_length - 1:actual_length,
                        :]
            if actual_length > 2:
                middle_indices = np.arange(1, actual_length - 1)
                selected_index = np.random.choice(middle_indices)
                middle_value = x_i_class[batch_index,
                               selected_index:selected_index + 1, :]
                adjusted_seq = np.concatenate(
                    (start_value, middle_value, end_value), axis=0)
            else:
                adjusted_seq = np.concatenate((start_value, end_value), axis=0)
            random_length = adjusted_seq.shape[0]
        else:
            # Choose a random length (throw an exception if min_length and max_length_for_sample are equal or min_length is greater)
            max_length_for_sample = max(min_length, actual_length - 1)
            if min_length >= max_length_for_sample:
                random_length = min_length
            else:
                random_length = np.random.randint(min_length,
                                                  max_length_for_sample + 1)
            random_lengths[i] = random_length

            # Sequence selection (always includes first and last values, and randomly selects from the middle)
            start_value = x_i_class[batch_index, 0:1, :]
            end_value = x_i_class[batch_index, actual_length - 1:actual_length,
                        :]
            num_middle_values = random_length - 2

            if num_middle_values > 0:
                possible_indices = np.arange(1, actual_length - 1)
                selected_indices = np.sort(
                    np.random.choice(possible_indices, num_middle_values,
                                     replace=False))
                middle_values = x_i_class[batch_index, selected_indices, :]
            else:
                middle_values = np.empty((0, feature_dim))

            adjusted_seq = np.concatenate(
                (start_value, middle_values, end_value), axis=0)

        augmented_sequences[i, :random_length, :] = adjusted_seq
        random_lengths[i] = random_length

    return augmented_sequences, random_lengths