with open('2017-10-10-12-19-14.csv', 'r') as f:
    labels = f.readline()
    labels = labels[1:]
    labels = labels.strip()
    labels = labels.split(',')
    print('number of label', len(labels))
for l in labels:
    print(len(l), l, type(l))

graph = ['40K', '4K', '1K', 'switch', 'pump']

for item in graph:
    print(item, type(item), item in labels)
