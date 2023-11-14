import multiprocessing

def worker_function(x):
    # Your worker function logic here
    return x * x

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4)  # Create a pool with 4 worker processes

    # Submit tasks to the pool
    results = pool.map(worker_function, list(range(10)))

    # Close the pool to prevent new tasks from being submitted
    pool.close()
    #results_2 = pool.map(worker_function, list(range(10)))
    # Block until all tasks are completed
    pool.join()  # Wait for all worker processes to finish

    # Now, you can safely process the results
    print(results)

