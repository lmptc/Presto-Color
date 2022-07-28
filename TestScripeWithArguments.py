import sys

_, a, b, c = sys.argv

a, b = int(a), int(b)
print('a is %d' % a)
print('b is %d' % b)
print(c)

print(list(c))

print('a/b is %f' % (a/b))
print(a/b)
