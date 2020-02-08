from multiprocessing import Pool, cpu_count


def progress_measurer(total_size):
    current_ten_percent = 10
    print("Started a job...")

    def measure(completed):
        nonlocal current_ten_percent, total_size
        if completed + 1 == total_size * current_ten_percent // 100:
            print("Finished {}%".format(current_ten_percent))
            current_ten_percent += 10

    return measure


def parallel_processing(func, array):
    pool = Pool(cpu_count())
    print("Processes pool created")
    results = pool.map(func, array)
    pool.close()
    pool.join()
    return results


def split_list(alist, parts=1):
    length = len(alist)
    return [alist[i * length // parts: (i + 1) * length // parts] for i in range(parts)]

