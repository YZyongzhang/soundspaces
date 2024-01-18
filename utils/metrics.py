def calculate_spl(successes, shortest_distances, path_lengths):
    """
    Calculate SPL (Success weighted by Path Length) with division by zero check.
    :param successes: List of binary success indicators (1 or 0).
    :param shortest_distances: List of shortest path distances.
    :param path_lengths: List of path lengths taken by the agent.
    :return: SPL value.
    """
    N = len(successes)
    spl_sum = 0
    for s, l, p in zip(successes, shortest_distances, path_lengths):
        print("p habdksufygblaiewygb", p)
        if p > 0:
            spl_sum += s * l / max(p, l)
        else:
            # Handle division by zero case
            spl_sum += 0  # Or some other value as per the specific requirements
    return spl_sum / N if N > 0 else 0

def calculate_soft_spl(shortest_distances, path_lengths):
    """
    Calculate soft-SPL with division by zero check.
    :param shortest_distances: List of shortest path distances.
    :param path_lengths: List of path lengths taken by the agent.
    :return: soft-SPL value.
    """
    N = len(shortest_distances)
    soft_spl_sum = 0
    for l, p in zip(shortest_distances, path_lengths):
        if p > 0:
            soft_spl_sum += min(p, l) / p * l / max(p, l)
        else:
            # Handle division by zero case
            soft_spl_sum += 0  # Or some other value as per the specific requirements
    return soft_spl_sum / N if N > 0 else 0
