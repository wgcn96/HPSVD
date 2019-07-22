

def hit_rate(re, test):
    def if_success(x):
        if x[0] >= re.shape[0]:
            return 0
        elif x[1] in re[x[0]]:
            return 1
        return 0

    return sum(test.apply(if_success, axis = 1))/test.shape[0]