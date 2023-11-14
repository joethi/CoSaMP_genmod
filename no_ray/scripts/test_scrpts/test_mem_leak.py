#this code simulates a memory leak
class MemoryLeakExample:
    def __init__(self):
        self.data = [0] * 1000000  # Allocating a large list

def simulate_memory_leak():
    # Create a list to hold instances of MemoryLeakExample
    instance_list = []

    # Simulate a memory leak by creating instances and never releasing them
    for _ in range(1000):
        instance = MemoryLeakExample()
        instance_list.append(instance)

    # At this point, we have created 1000 instances and they are still in memory

if __name__ == "__main__":
    simulate_memory_leak()
import pdb; pdb.set_trace()

