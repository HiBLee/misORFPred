from Bio import SeqIO
import numpy as np

def ESKmer(fastas, k=1):
    NA = 'ACGT'
    encodings = []
    if k == 1:
        NA_mers = NA
    if k == 2:
        NA_mers = [NA1 + NA2 for NA1 in NA for NA2 in NA]
    if k == 3:
        NA_mers = [NA1 + NA2 + NA3 for NA1 in NA for NA2 in NA for NA3 in NA]
    if k == 4:
        NA_mers = [NA1 + NA2 + NA3 + NA4 for NA1 in NA for NA2 in NA for NA3 in NA for NA4 in NA]
    if k == 5:
        NA_mers = [NA1 + NA2 + NA3 + NA4 + NA5 for NA1 in NA for NA2 in NA for NA3 in NA for NA4 in NA for NA5 in NA]
    if k == 6:
        NA_mers = [NA1 + NA2 + NA3 + NA4 + NA5 + NA6 for NA1 in NA for NA2 in NA for NA3 in NA for NA4 in NA for NA5 in NA for NA6 in NA]
    if k == 7:
        NA_mers = [NA1 + NA2 + NA3 + NA4 + NA5 + NA6 + NA7 for NA1 in NA for NA2 in NA for NA3 in NA for NA4 in NA for NA5 in NA for NA6 in NA for NA7 in NA]
    if k == 8:
        NA_mers = [NA1 + NA2 + NA3 + NA4 + NA5 + NA6 + NA7 + NA8 for NA1 in NA for NA2 in NA for NA3 in NA for NA4 in NA for NA5 in NA for NA6 in NA for NA7 in NA for NA8 in NA]

    header = ['#']

    for NA in NA_mers:
        header.append('ESKmer_' + str(k) + '_' + NA)
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], i[1]
        sequence = sequence.replace('U', 'T')
        # sequence = sequence.replace('T', 'U')
        code = [name]
        myDict = {}
        for mer in NA_mers:
            myDict[mer] = 0
        sum = 0

        if k == 1:
            for index in range(len(sequence)):
                myDict[sequence[index]] += 1
                sum += 1

        if k != 1:
            if k % 2 == 0:
                median = int(k / 2)
            else:
                median = int((k + 1) / 2)

            for index1 in range(len(sequence) - k + 1):
                tuple1 = sequence[index1: index1 + median]
                for index2 in range(index1 + median, len(sequence) - (k - median - 1)):
                    tuple2 = sequence[index2: index2 + (k - median)]
                    myDict[tuple1 + tuple2] += 1
                    sum += 1

        for tuple_pair in NA_mers:
            code.append(myDict[tuple_pair] / sum)
        encodings.append(code)

    return encodings

def ReadFileFromFasta(filepath):
    seq = []
    for seq_record in SeqIO.parse(filepath, "fasta"):
        seq.append(['>' + seq_record.id.strip(), str(seq_record.seq).strip()])
    return seq

def FeatureGenerator(fastas):
    labels = []

    FeatureDict = {}
    FeatureNameDict = {}

    for i in fastas:
        name = i[0]
        if str(name).startswith('>') and str(name).find('_URS') != -1:
            labels.append(0)
        else:
            labels.append(1)

    ESMER3 = np.array(ESKmer(fastas, 3))
    ESMER4 = np.array(ESKmer(fastas, 4))
    ESMER5 = np.array(ESKmer(fastas, 5))

    FeatureDict['ESMER3'] = np.array(ESMER3[1:, 1:], dtype=float)
    FeatureNameDict['ESMER3'] = ESMER3[:1, :][0]

    FeatureDict['ESMER4'] = np.array(ESMER4[1:, 1:], dtype=float)
    FeatureNameDict['ESMER4'] = ESMER4[:1, :][0]

    FeatureDict['ESMER5'] = np.array(ESMER5[1:, 1:], dtype=float)
    FeatureNameDict['ESMER5'] = ESMER5[:1, :][0]

    return FeatureDict, np.array(labels).astype(int), FeatureNameDict