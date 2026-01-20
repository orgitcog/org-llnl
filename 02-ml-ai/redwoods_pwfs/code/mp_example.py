from multiprocessing import Pool

def add_numbers(a, b):
    return a + b

if __name__ == '__main__':
    data = [(1, 2), (3, 4), (5, 6)]
    with Pool() as pool:
        results = pool.starmap(add_numbers, data)
    print(results)
