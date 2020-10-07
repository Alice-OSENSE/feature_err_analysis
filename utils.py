import scipy.io


def read_mat(path):
    mat = scipy.io.loadmat(path)
    sorted_keys = sorted(mat.keys())
    print(sorted_keys)
    return mat


def get_pmapxy(path):
    mat = read_mat(path)
    return mat['dmap'][0][0][0]