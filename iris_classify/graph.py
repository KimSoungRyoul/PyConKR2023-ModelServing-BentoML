import matplotlib.pyplot as plt
import numpy as np

batches = ['500', '1000', '2000']
origin_avg = [0.013489195999999998, 0.01884337000000001, 0.023584341999999994]
dis_runner_avg = [0.013557794000000001, 0.015433535999999998, 0.020371260000000016]
origin_median = [0.0116975, 0.018504, 0.0242135]
dis_runner_median = [0.011887, 0.013708999999999999, 0.0215655]

bar_width = 0.2

r1 = np.arange(len(batches))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

#plt.bar(r1, origin_avg, color='b', width=bar_width, edgecolor='white', label='origin AVG')
#plt.bar(r2, dis_runner_avg, color='r', width=bar_width, edgecolor='white', label='dis-runner AVG')
plt.bar(r3, origin_median, color='b', width=bar_width, edgecolor='white', label='origin Median')
plt.bar(r4, dis_runner_median, color='r', width=bar_width, edgecolor='white', label='dis-runner Median')

plt.xlabel('Median Latency By Batch Size', fontweight='bold', fontsize=15)
plt.ylabel('Value', fontweight='bold', fontsize=15)
plt.xticks([r + bar_width * 1.5 for r in range(len(batches))], batches)
plt.legend()

#plt.show()
plt.savefig("./iris_distributed_median_performance_by_batch_size.png")
#plt.savefig("./iris_distributed_median_performance_by_batch_size.png")


# import matplotlib.pyplot as plt
# import numpy as np
#
# batches = ['500', '1000', '2000']
#
# percentiles_list_by_batch_size = [
#     # batch size:500, percentilas_50
#     [0.011701, 0.011895],
#     # batch size:500, percentilas_75
#     [0.016609, 0.016824],
#     # batch size:500, percentilas_100
#     [0.06671, 0.040127],
#     # batch size:1000, percentilas_50
#     [0.01852, 0.013711],
#     # batch size:1000, percentilas_75
#     [0.019785, 0.01771],
#     # batch size:1000, percentilas_100
#     [0.112543, 0.046506],
#     #  batch size:2000, percentilas_50
#     [0.024216, 0.021569],
#     #  batch size:2000, percentilas_75
#     [0.02479, 0.022553],
#     #  batch size:2000, percentilas_100
#     [0.047753, 0.04615],
# ]
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# percentiles_list_by_batch_size = [
#     [0.011701, 0.011895],
#     [0.016609, 0.016824],
#     [0.06671, 0.040127],
#     [0.01852, 0.013711],
#     [0.019785, 0.01771],
#     [0.112543, 0.046506],
#     [0.024216, 0.021569],
#     [0.02479, 0.022553],
#     [0.047753, 0.04615],
# ]
#
# labels = ['50%', '75%', '100%']
# batches = ['500', '1000', '2000']
#
# bar_width = 0.35
# index = np.arange(len(labels))
#
# for i, batch in enumerate(batches):
#     plt.subplot(1, len(batches), i+1)
#     percentiles = percentiles_list_by_batch_size[i*len(labels):(i+1)*len(labels)]
#     plt.bar(index, [p[0] for p in percentiles], bar_width, label='origin')
#     plt.bar(index + bar_width, [p[1] for p in percentiles], bar_width, label='dis-runner')
#     plt.xlabel('Percentiles')
#     plt.ylabel('Latency')
#     plt.title('Batch Size: ' + batch)
#     plt.xticks(index + bar_width / 2, labels)
#     plt.legend()
#
# plt.tight_layout()
# #plt.show()
# plt.savefig("./iris_distributed_percentiles_performance_by_batch_size.png")