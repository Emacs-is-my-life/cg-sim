import math

def KB_to_num_pages(memory_size_KB, page_size_KB: int = 4):
    return math.ceil(memory_size_KB / page_size_KB)
