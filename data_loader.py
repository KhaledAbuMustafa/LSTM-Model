import numpy as np

class DataLoader:

    def read_data_from_file(self, file_path):
        x_values = []
        y_values = []

        with open(file_path, 'r') as file:
            for line in file:
                columns = line.strip().split()
                x_micro = float(columns[0])
                y = float(columns[1])
                x_values.append(x_micro)
                y_values.append(y)

        return np.array(x_values), np.array(y_values)