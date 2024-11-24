import numpy as np
np.random.seed(28)

def generate_bits(num_sequences = 100000,min_length=1,max_length=16):
    sequences = []
    labels = []

    for _ in range(num_sequences):
        seq_length = np.random.randint(min_length, max_length + 1)  
        sequence = np.random.randint(0, 2, seq_length).tolist()  
        count_of_ones = sum(sequence) 
        sequences.append(sequence)
        labels.append(count_of_ones)


    max_seq_length = max_length
    bits_padded = np.array([np.pad(seq, (0,max_seq_length - len(seq)), 'constant', constant_values=0) for seq in sequences])
    labels = np.array(labels)
    print(bits_padded.shape)
    print(f"Padded Sequences (First 3): \n{bits_padded[:3]}")
    print(f"Labels (First 3): \n{labels[:3]}")
    bits_padded = bits_padded.reshape((num_sequences,max_length,1))
    labels =  labels.reshape((num_sequences,1))

    print(f"Example from dataset: {bits_padded[0]} -> Count of '1's: {labels[0]}")
    return bits_padded,labels

bits_padded,labels = generate_bits()
np.save("./data/interim/5/4/bits_padded.npy",bits_padded)
np.save("./data/interim/5/4/labels.npy",labels)


