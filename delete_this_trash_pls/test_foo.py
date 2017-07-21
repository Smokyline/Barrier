
def try_test():
    r = 0
    try:
        r = r + (1/0)
    finally:
        return r


print(try_test())