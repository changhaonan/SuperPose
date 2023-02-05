# randint input
# out 123 -> 321
import math


def reverse_int(num):
    # checking
    assert(num >= 0)
    # parse it into a list
    digit_list = []
    while True:
        digit = math.floor(num / 10)
        num = num - digit * 10
        digit_list.append(digit)
        if num < 10:
            digit_list.append(num)
            break
    # reverse it
    reverse_int = 0
    num_digit = len(digit_list)
    for i in range(num_digit):
        reverse_int += digit_list[i] * math.pow(10, i)
    return reverse_int



if __name__ == "__main__":
    import random
    N = 10
    random_list = []
    for i in range(N):
        random_list.append(random.randint(1, 100))

    inverse_random_list = []
    for r in random_list:
        inverse_random_list.append(reverse_int(r))

    # Eval
    for n, inv_n in zip(random_list, inverse_random_list):
        print(f"Before: {n}, After: {inv_n}.\n")