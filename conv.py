import math
from typing import Sequence


ConvInput = tuple[Sequence, Sequence]
ConvResult = list[Sequence[tuple[int, int]]]


# This number counts how many operations were performed
total_operations = 0


def naive_polynomial_multiplication(w: Sequence[int], v: Sequence[int]) -> ConvResult:
    """This code does polynomial multiplication between two vectors w,v representing
    polynomials.

    Args:
        w (Sequence[int]): First vector
        v (Sequence[int]): Second vector

    Returns:
        ConvT: The convolution, always of size len(w)+len(v)-1
    """
    count_conv_operations(w, v)

    # Do convolution WITHOUT FFT
    pairs = []
    for n in range(len(w) + len(v)):
        current = []
        for i in range(min(n + 1, len(w))):
            j = n - i
            if j >= 0 and j < len(v):
                current.append((w[i], v[j]))
        if len(current) > 0:
            pairs.append(current)
    return pairs


# Until we upload the correct one.
FFT = naive_polynomial_multiplication


def count_conv_operations(a: Sequence[int], b: Sequence[int]) -> None:
    global total_operations
    maxl = max(len(a), len(b))
    minl = min(len(a), len(b))
    total_operations += int(math.ceil(maxl * math.log2(minl)))


def sub_convolutions(sub_from: ConvResult, right: ConvResult) -> ConvResult:
    """Given two convolution results, subtract the results of the second from the first,
    meaning that if an operation was done in both convolutions

    Args:
        sub_from (ConvT): The convolution to sub from
        right (ConvT): What to sub
    """
    # Assume both are sorted.
    sub_from = sub_from[:]
    start_center = sum(right[0][0])
    for i in range(len(right)):
        rprime = set([(x[1], x[0]) for x in right[i]])
        sub = rprime | set(right[i])

        temp = list(set(sub_from[i + start_center]) - sub)
        # Ensure we didn't remove extra elements
        assert len(temp) == len(sub_from[i+start_center]) - len(sub)
        temp.sort(key=lambda x: x[0])
        sub_from[i + start_center] = temp
    return sub_from


def print_conv(conv_result: ConvResult, min_len: int = 0) -> None:
    """Prints convolution result

    Args:
        res (ConvT): The convolution result
        min_len (int, optional): Minimal length of a result to print. Defaults to 0.
    """
    for elem in conv_result:
        if len(elem) >= min_len:
            x, y = elem[0]
            palindromic_center = x + y
            print(f"{len(elem):02d} {palindromic_center:02d} {elem}")


def print_conv_stats(res: ConvResult, size: int) -> None:
    """Prints the statistics of a given convolution

    Args:
        res (ConvT): The resulting convolution
        size (int): The original size of the convolution vector 
        (assuming multiplying the vector unto itself)
    """
    max_len = len(max(res, key=lambda l: len(l)))
    expected = int(size * (math.log2(size) ** 2)) // size
    print(f"{max_len=}, {total_operations=}, {expected=}, Operations per element: {total_operations//size}")


def print_conv_stats_fast(size: int) -> None:
    """Prints the statistics of a given convolution simulation

    Args:
        size (int): The original size of the convolution vector
    """
    logn = math.ceil(math.log2(size))
    total_per_elem = ((logn - 2) * (logn - 1)) // 2
    print(f"{total_operations=}, {total_per_elem=}, Operations per element: {total_operations//size}. Ratio: {total_per_elem * size / total_operations:.2f}")


def generate_next_step(lo_range: range, hi_range: range) -> tuple[ConvInput, ConvInput]:
    """Creates the next convolution step, meaning the next convolution to reduce 
    from the original result.

    Args:
        lo_range (range): Previous low range sub convolution
        hi_range (range): Previous high range sub convolution

    Returns:
        tuple[ConvInput, ConvInput]: The next pair of convolutions to perform
    """
    size = len(lo_range)
    lo = lo_range.start
    hi = hi_range.start
    next_size = size >> 1
    return ((range(lo, lo + next_size), range(hi - next_size, hi)),
            (range(lo + size, lo + size + next_size), range(hi + next_size, hi + size)))


def generate_steps(initial_size: int) -> list[list[ConvInput]]:
    """Generates all reducing convolutions for a given array

    Args:
        initial_size (int): The original size of the array

    Returns:
        list[list[tuple[range, range]]]: All convolutions that need to be performed.
    """
    size = initial_size >> 2
    left = size
    right = initial_size - left
    # Always the first step
    result = []
    result.append(((range(left), range(right, right + size)),))
    while size > 2:
        current = []
        for res in result[-1]:
            current.extend(generate_next_step(res[0], res[1]))
        result.append(current)
        size >>= 1
    return result


def simulate_for_exponent(exponent: int):
    global total_operations
    total_operations = 0
    size = (1 << exponent)
    first = (range(size), range(size))
    convolutions = generate_steps(size)

    count_conv_operations(first[0], first[1])
    for conv_step in convolutions:
        for step in conv_step:
            count_conv_operations(step[0], step[1])

    print_conv_stats_fast(size)


def animate_for_exponent(exponent: int):
    global total_operations
    total_operations = 0
    size = (1 << exponent)
    first = (range(size), range(size))
    convolutions = generate_steps(size)

    result = naive_polynomial_multiplication(first[0], first[1])
    yield result
    for conv_step in convolutions:
        for step in conv_step:
            result = sub_convolutions(
                result, naive_polynomial_multiplication(step[0], step[1]))
        yield result



def stable_convolution(w: int):
    reduction_steps = generate_steps(w)

    result = FFT(range(w), range(w)) # Not reversed, because we are interested in palindromes.
    for conv_step in reduction_steps:
        for step in conv_step:
            result = sub_convolutions(result, FFT(step[0], step[1]))
    return result




def run_for_exponent(exponent: int):
    res = None
    for res in animate_for_exponent(exponent):
        # Currently, do nothing
        pass

    if res:
        print_conv_stats(res, 1 << exponent)


if __name__ == "__main__":
    run_for_exponent(7)         # Input of size 128
    run_for_exponent(10)        # Input of size 1024
    simulate_for_exponent(10)   # Input of size 1024
    simulate_for_exponent(16)   # Input of size 65000
    simulate_for_exponent(20)   # Input of size ~1M
