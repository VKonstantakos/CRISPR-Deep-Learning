def most_common(lst):
    '''
    Find the most common value of a list.

    :parameter lst: list to analyse.
    :return: most common value.
    '''

    return max(set(lst), key=lst.count)


def find_pam(seq):
    '''
    Find the PAM index of a given column containing sequences.

    :seq: column with sequences to analyse.
    :return: PAM index (if all PAMs' indices match).
    '''

    # Initialize lists of PAM indices
    agg = tgg = cgg = ggg = []

    # Find indices for each PAM
    for sequence in seq:
        agg_index = sequence.find('AGG')
        agg.append(agg_index)

        tgg_index = sequence.find('TGG')
        tgg.append(tgg_index)

        cgg_index = sequence.find('CGG')
        cgg.append(cgg_index)

        ggg_index = sequence.find('GGG')
        ggg.append(ggg_index)

    # Remove not found (-1) values
    agg_new = [x for x in agg if x != -1]
    tgg_new = [x for x in tgg if x != -1]
    cgg_new = [x for x in cgg if x != -1]
    ggg_new = [x for x in ggg if x != -1]

    # Find the most common index and check if it's the same for all PAMs
    if most_common(agg_new) == most_common(tgg_new) == most_common(cgg_new) == most_common(ggg_new):
        return most_common(agg_new)
    else:
        raise Exception('PAM indices are different')


def extract_koike_yusa(ds, seq_col=5, rep1_col=6, rep2_col=7):
    '''
    Process Koike-Yusa dataset and extract 23-nt and 30-nt sequences.
    Get the log fold change of both replicates, inverse the sign due to negative selection screening
    (so that higher values mean higher efficiency), and calculate their mean.

    :parameter ds: dataset to analyse i.e. Koike-Yusa as provided by Xu.
    :parameter seq_col: # position of column containing extended sequences e.g. 5th column -> 5.
    :parameter rep1_col: # position of column containing log fold change of the first replicate e.g. 6th column -> 6.
    :parameter rep2_col: # position of column containing log fold change of the second replicate e.g. 7th column -> 7.
    :return: 23-nt sequences, 30-nt sequences and the mean log fold change.
    '''

    # Get the extended sequence and convert it into uppercase characters
    seq = ds.iloc[:, seq_col-1].str.upper()

    # Find the PAM index
    pam_index = find_pam(seq)

    # Extract the 23-nt and 30-nt sequences according to the PAM index
    seq_23 = seq.str[pam_index-20:pam_index+3]
    seq_30 = seq.str[pam_index-24:pam_index+6]

    # Get the log fold change of the two replicates and inverse the sign (due to negative selection screen)
    log_eff1 = - (ds.iloc[:, rep1_col-1])
    log_eff2 = - (ds.iloc[:, rep2_col-1])

    # Calculate mean log fold change
    log_eff_mean = (log_eff1 + log_eff2)/2

    return seq_23, seq_30, log_eff_mean


def extract_haeussler(ds, data_name, data_col=1, seq_col=3, ext_seq_col=7, eff_col=4):
    '''
    Extracts 23-nt and 30-nt sequences with their respective efficiencies from Haeussler dataset.

    :parameter ds: dataset to analyse i.e. Haeussler.
    :parameter data_name: the name of the dataset we want to extract.
                          We used: 'doench2016_hg19', 'Hct116', '293T', 'doench2014-Mm', Hl60, 'shkumatava'.
    :parameter data_col: # position of column containing the datasets' names e.g. 1st column -> 1.
    :parameter seq_col: # position of column containing spacer (20-nt or 23-nt) sequences e.g. 3rd column -> 3.
    :parameter ext_seq_col: # position of column containing extended sequences e.g. 7th column -> 7.
    :parameter eff_col: # position of column containing efficiencies e.g. 4th column -> 4.
    :return: lists of 23-nt, 30-nt sequences and their efficiencies.
    '''

    # Extract the relevant dataset by name ('doench2016_hg19', 'Hct116', '293T', 'doench2014-Mm', Hl60, 'shkumatava')
    ds_new = ds[ds.iloc[:, data_col-1].str.contains(data_name)]

    # Initialize lists to obtain 23-nt and 30-nt sequences with their efficiencies
    seq_23 = []
    seq_30 = []
    eff = []

    # Iterate rows to find starting index and extract the appropriate sequences and their corresponding efficiencies
    for _, row in ds_new.iterrows():
        start_index = row.iloc[ext_seq_col-1].find(row.iloc[seq_col-1])
        seq_23.append(row.iloc[ext_seq_col-1][start_index:start_index + 23])
        seq_30.append(row.iloc[ext_seq_col-1][start_index-4:start_index + 26])

        # Inverse the sign of the knock-out efficiencies from Wang/Xu dataset (HL60) due to negative selection screening
        if data_name == 'Hl60':
            eff.append(-row.iloc[eff_col-1])

        # Convert percent value to decimal for Shkumatava dataset (Zebrafish) due to % indels induced at target sequence
        elif data_name == 'shkumatava':
            eff.append(row.iloc[eff_col - 1] / 100)

        else:
            eff.append(row.iloc[eff_col-1])

    return seq_23, seq_30, eff


def extract_shalem(ds, seq_col=1, ext_seq_col=2, eff_col=12):
    '''
    Extracts 23-nt and 30-nt sequences with their respective efficiencies from Shalem dataset.

    :parameter ds: dataset to analyse i.e. Shalem as provided by Doench.
    :parameter seq_col: # position of column containing spacer (20-nt) sequences e.g. 1st column -> 1.
    :parameter ext_seq_col: # position of column containing extended sequences e.g. 2nd column -> 2.
    :parameter eff_col: # position of column containing efficiencies e.g. 12th column -> 12.
    :return: lists of 23-nt, 30-nt sequences and their efficiencies.
    '''

    # Initialize lists to obtain 23-nt and 30-nt sequences with their efficiencies
    seq_23 = []
    seq_30 = []
    eff = []

    # Iterate rows to find starting index and extract the appropriate sequences and their corresponding efficiencies
    for _, row in ds.iterrows():
        start_index = row.iloc[ext_seq_col-1].find(row.iloc[seq_col-1])
        seq_23.append(row.iloc[ext_seq_col-1][start_index:start_index + 23])
        seq_30.append(row.iloc[ext_seq_col-1][start_index-4:start_index + 26])
        eff.append(row.iloc[eff_col-1])

    return seq_23, seq_30, eff


def rescale(ds, eff_col=1):
    '''
    Rescale all efficiencies into range [0,1].

    :parameter ds: dataset to analyse.
    :parameter eff_col: # position of column containing efficiencies e.g. 1st column -> 1.
    :return: column containing rescaled efficiencies.
    '''

    y = ds.iloc[:, eff_col-1]
    return (y - y.min()) / (y.max() - y.min())


def csv_to_fasta(ds, length, seq_col=2):
    '''
    Extract and write 20-nt/30-nt sequences into a FASTA file to be used for E-CRISP/DeepCas9, respectively.

    :parameter ds: dataset to analyse.
    :parameter length: length of extracted sequence (20 for E-CRISP, 30 for DeepCas9).
    :parameter seq_col: # position of column containing sequences e.g. 2nd column -> 2.
    :return: None.
    '''

    if length == 20:
        fasta = open('E-CRISP.fasta', 'w+')

        for index, row in ds.iterrows():
            fasta.write('>Seq_' + str(index+1) + '\n' +
                        row.iloc[seq_col-1][4:24] + '\n')

        fasta.close()

    elif length == 30:
        fasta = open('DeepCas9.fasta', 'w+')

        for index, row in ds.iterrows():
            fasta.write('>Seq_' + str(index+1) + '\n' +
                        row.iloc[seq_col-1] + '\n')

        fasta.close()

    else:
        raise Exception('Length should be either 20 or 30 nucleotides.')
