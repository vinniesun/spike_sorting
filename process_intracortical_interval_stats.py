import csv
import numpy as np
import json

if __name__ == "__main__":
    class1_stats, class2_stats, class3_stats = dict(), dict(), dict()
    for spk_class in [1, 2, 3]:
        for difficulty in ["Easy1", "Easy2", "Difficult1", "Difficult2"]:
            for noise_level in ["005", "01", "015", "02"]:
                filename = f"C_{difficulty}_noise{noise_level}.mat_{spk_class}.csv"
                filepath = "./interval_statistics/"
                complete_filename = filepath + filename
                # print(f"Reading file: {complete_filename}")

                with open(complete_filename, mode='r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # skip the header
                    for i, row in enumerate(reader):
                        key = f"{row[0]},{row[1]}"
                        if spk_class == 1:
                            if key not in class1_stats:
                                class1_stats[key] = int(row[2])
                            else:
                                class1_stats[key] += int(row[2])
                        elif spk_class == 2:
                            if key not in class2_stats:
                                class2_stats[key] = int(row[2])
                            else:
                                class2_stats[key] += int(row[2])
                        else:
                            if key not in class3_stats:
                                class3_stats[key] = int(row[2])
                            else:
                                class3_stats[key] += int(row[2])

    # Save the stats to JSON files
    # with open('./interval_statistics/class1_stats.json', 'w') as f:
    #     sorted_class1_stats = dict(sorted(class1_stats.items(), key=lambda item: item[1], reverse=True))
    #     json.dump(sorted_class1_stats, f, indent=4)
    # with open('./interval_statistics/class2_stats.json', 'w') as f:
    #     sorted_class2_stats = dict(sorted(class2_stats.items(), key=lambda item: item[1], reverse=True))
    #     json.dump(sorted_class2_stats, f, indent=4)
    # with open('./interval_statistics/class3_stats.json', 'w') as f:
    #     sorted_class3_stats = dict(sorted(class3_stats.items(), key=lambda item: item[1], reverse=True))
    #     json.dump(sorted_class3_stats, f, indent=4)

    # save the stats to csv files
    with open('./interval_statistics/class1_stats.csv', 'w', newline='\n') as f:
        sorted_class1_stats = dict(sorted(class1_stats.items(), key=lambda item: item[1], reverse=True))
        writer = csv.writer(f)
        writer.writerow(['t1', 't2', 'Count'])
        for key, value in sorted_class1_stats.items():
            intervals = key.split(',')
            writer.writerow([intervals[0], intervals[1], value])
    with open('./interval_statistics/class2_stats.csv', 'w', newline='\n') as f:
        sorted_class2_stats = dict(sorted(class2_stats.items(), key=lambda item: item[1], reverse=True))
        writer = csv.writer(f)
        writer.writerow(['t1', 't2', 'Count'])
        for key, value in sorted_class2_stats.items():
            intervals = key.split(',')
            writer.writerow([intervals[0], intervals[1], value])
    with open('./interval_statistics/class3_stats.csv', 'w', newline='\n') as f:
        sorted_class3_stats = dict(sorted(class3_stats.items(), key=lambda item: item[1], reverse=True))
        writer = csv.writer(f)
        writer.writerow(['t1', 't2', 'Count'])
        for key, value in sorted_class3_stats.items():
            intervals = key.split(',')
            writer.writerow([intervals[0], intervals[1], value])
