import numpy as np

barrier_vars = ['B_oneVoneP_Y']
res_path = '/Users/Ivan/Documents/workspace/result/Barrier/comp/'
print('path:%s\nmods: %s\n' % (res_path, barrier_vars))

alg_res = np.empty((0, 5))

for key in barrier_vars:
    print(key, end='... ')
    try:
        txt_file = open(res_path + key + '.txt')
        for line in txt_file:
            parced_line = line.split(' ')
            alg_res = np.append(alg_res, [parced_line], axis=0)
        print('read')
    except Exception as e:
        # print('no %s in folder %s' % (key, res_path))
        print(e)


def sort_results(res, acc=False):
    if acc:
        srt = np.argsort(res)[::-1]
    else:
        srt = np.argsort(res)
    sorted_res = res[srt]
    place_in_array = np.zeros((1, len(res)))[0]

    mean_place = []
    last_value = None

    def append_mean_idx_to_arr(mean_pl):
        mns = np.mean(mean_pl)
        for j in mean_place:
            place_in_array[j - 1] = mns

    for place, res_value in enumerate(sorted_res):
        mean_place.append(place)

        if res_value != last_value:
            append_mean_idx_to_arr(mean_place)
            place_in_array[place] = np.mean(mean_place)
            mean_place = []

        last_value = res_value

    final_arr = []
    for idx in range(len(res)):
        pl_idx = np.where(srt == idx)[0]
        final_arr.append(place_in_array[pl_idx])

    return np.array(final_arr).T[0]


acc_res = np.array(alg_res[:, 2]).astype(float)
sqr_res = np.array(alg_res[:, 3]).astype(float)

s_acc_res = sort_results(acc_res, acc=True)
s_sqr_res = sort_results(sqr_res, acc=False)

# print(np.array([(a, sa) for a, sa in zip(acc_res, s_acc_res)]))

a = 0.48
p = 0.52
s_sum_res = s_acc_res * a + s_sqr_res * p

min_sort = np.argsort(s_sum_res)
print(alg_res[min_sort, 1:])
