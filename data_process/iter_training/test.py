def f(x, count):
    count[0] += 1  # 每次调用 f 时，增加一次计数
    if x <= 2:
        return 1
    return f(x - 3, count) * f(x - 5, count)

count = [0]
f(17, count)
print(f"Total calls: {count[0]}")