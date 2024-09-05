import csv

rows = []

with open('aggregate_data.csv', 'r') as file:
    reader = csv.reader(file)
    for i,row in enumerate(reader):
        # if i == 0:
        #     rows.append(row)
        #     print(row)
        if i % 50 == 0:
            rows.append(row)
            print(row)
            
        # print(row)
        
with open('aggregate_data_sample.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(rows)