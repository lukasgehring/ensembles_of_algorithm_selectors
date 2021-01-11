def read_file(path):
    with open(path, 'r') as f:
        return f.read()

f1 = read_file('pred.txt')
f2 = read_file('pred_b.txt')

f1 = f1.replace('\n', '')
f1 = ' '.join(f1.split())
f1 = f1.split('][')

f2 = f2.replace('\n', '')
f2 = ' '.join(f2.split())
f2 = f2.split('][')

count = 0
for line1, line2 in zip(f1, f2):
    line1 = line1.replace('[', '').replace(']', '')
    line2 = line2.replace('[', '').replace(']', '')
    line1 = ' '.join(line1.split())
    line2 = ' '.join(line2.split())
    if line1 != line2:
        print(count)
        print(line1)
        print(line2)
    count = count + 1

print(f1[69:73])
print(f2[73])