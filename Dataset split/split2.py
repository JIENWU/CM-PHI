import numpy as np
import pandas as pd
import csv
import math
import random
csv.field_size_limit(500 * 1024 * 1024)


def read_csv(save_list, file_name):
    csv_reader = csv.reader(open(file_name))
    for row in csv_reader:
        save_list.append(row)
    return


def store_csv(data, file_name):
    with open(file_name, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)
    return


def generate_negative_sample(relationship_pd):
    relationship_matrix = pd.pivot_table(relationship_pd, index='Pair1', columns='Pair2', values='Rating', fill_value=0)
    negative_sample = []
    counter = 0
    while counter < len(relationship_pd):
        print(counter)
        temp_1 = random.randint(0, len(relationship_matrix.index) - 1)
        temp_2 = random.randint(0, len(relationship_matrix.columns) - 1)
        if relationship_matrix.iloc[temp_1, temp_2] == 0:
            relationship_matrix.iloc[temp_1, temp_2] = -1
            row = []
            row.append(np.array(relationship_matrix.index).tolist()[temp_1])
            row.append(np.array(relationship_matrix.columns).tolist()[temp_2])
            negative_sample.append(row)

            counter = counter + 1

        else:
            pass

    return negative_sample, relationship_matrix


if __name__ == '__main__':

    relationship = []
    read_csv(relationship, 'PBI pairs.csv')
    random.shuffle(relationship)    # random shuffle
    relationship_train = relationship[0: int(0.7 * len(relationship))]
    store_csv(relationship_train, 'PositiveSample_Train_PBI.csv')
    relationship_validation = relationship[int(0.7 * len(relationship)):int(0.8 * len(relationship))]
    store_csv(relationship_validation, 'PositiveSample_Validation_PBI.csv')
    relationship_test = relationship[int(0.8 * len(relationship)):]
    store_csv(relationship_test, 'PositiveSample_Test_PBI.csv')

    relationship_pd = pd.DataFrame(relationship, columns=['Pair1', 'Pair2'])
    relationship_pd['Rating'] = [1] * len(relationship_pd)
    negative_sample, relationship_matrix = generate_negative_sample(relationship_pd)
    relationship_matrix.to_csv('PBI_Relationship_Matrix.csv')

    store_csv(negative_sample, 'NegativeSample_all_PBI.csv')
    negative_sample_train = negative_sample[0: int(0.7 * len(negative_sample))]
    store_csv(negative_sample_train, 'NegativeSample_Train_PBI.csv')
    negative_sample_validation = negative_sample[int(0.7 * len(negative_sample)):int(0.8 * len(negative_sample))]
    store_csv(negative_sample_validation, 'NegativeSample_Validation_PBI.csv')
    negative_sample_test = negative_sample[int(0.8 * len(negative_sample)):]
    store_csv(negative_sample_test, 'NegativeSample_Test_PBI.csv')


