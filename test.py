import numpy as np

list = [0, 1, 4, 7, 8]
remove_id_list = [4, 7]
remove_index = np.where(np.array(list) == np.array(remove_id_list)[:, None])[-1]

print(1)