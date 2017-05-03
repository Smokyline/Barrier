

path = '/Users/Ivan/Documents/workspace/resources/csv/Barrier/kvz/gridVers/d0.1/param_str.txt'

f = open(path, 'r')
f_str = f.read()
for s in ['[', ']', "'", ',']:
    print(s)
    f_str = f_str.replace(s, '')
f_list = f_str.split()
print(type(f_list[0]))