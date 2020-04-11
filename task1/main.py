

def findMaxSubArray(A):
    """
    Returns continuous subarray of A array with the maximum sum of elements
    :param A: array of integers, length(A) > 0
    :return: array of integers, length(output) > 0
    """
    max_sum = A[0]
    max_sum_borders = 0, 1
    sums = A[:1]

    def check_candidate_sum(candidate_sum, borders):
        nonlocal max_sum
        nonlocal max_sum_borders
        if candidate_sum > max_sum:
            max_sum = candidate_sum
            max_sum_borders = borders

    for i in range(1, len(A)):
        current_sum = sums[-1]+A[i]
        sums.append(current_sum)
        check_candidate_sum(current_sum, (0, i+1))
        for j in range(i):
            candidate_sum = current_sum - sums[j]
            if candidate_sum > max_sum:
                check_candidate_sum(candidate_sum, (j+1, i+1))
    return A[max_sum_borders[0]:max_sum_borders[-1]]
