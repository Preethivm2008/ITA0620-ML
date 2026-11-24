# FIND-S Algorithm

import csv

data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

hypothesis = ['0'] * (len(data[0]) - 1)

for row in data:
    if row[-1] == 'Yes':
        for i in range(len(hypothesis)):
            if hypothesis[i] == '0':
                hypothesis[i] = row[i]
            elif hypothesis[i] != row[i]:
                hypothesis[i] = '?'

print("Final Hypothesis:", hypothesis)