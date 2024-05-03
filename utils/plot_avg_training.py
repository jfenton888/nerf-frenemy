
import numpy as np
import matplotlib.pyplot as plt

# Read the file "ablation_training_loss.csv" which has columns "Step","PROPS-NeRF-ablate-19 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-19 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-19 - Train Loss Dict/original_segmentation_loss__MAX","PROPS-NeRF-ablate-18 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-18 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-18 - Train Loss Dict/original_segmentation_loss__MAX","PROPS-NeRF-ablate-17 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-17 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-17 - Train Loss Dict/original_segmentation_loss__MAX","PROPS-NeRF-ablate-16 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-16 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-16 - Train Loss Dict/original_segmentation_loss__MAX","PROPS-NeRF-ablate-15 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-15 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-15 - Train Loss Dict/original_segmentation_loss__MAX","PROPS-NeRF-ablate-14 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-14 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-14 - Train Loss Dict/original_segmentation_loss__MAX","PROPS-NeRF-ablate-13 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-13 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-13 - Train Loss Dict/original_segmentation_loss__MAX","PROPS-NeRF-ablate-12 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-12 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-12 - Train Loss Dict/original_segmentation_loss__MAX","PROPS-NeRF-ablate-11 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-11 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-11 - Train Loss Dict/original_segmentation_loss__MAX","PROPS-NeRF-ablate-10 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-10 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-10 - Train Loss Dict/original_segmentation_loss__MAX","PROPS-NeRF-ablate-9 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-9 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-9 - Train Loss Dict/original_segmentation_loss__MAX","PROPS-NeRF-ablate-8 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-8 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-8 - Train Loss Dict/original_segmentation_loss__MAX","PROPS-NeRF-ablate-7 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-7 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-7 - Train Loss Dict/original_segmentation_loss__MAX","PROPS-NeRF-ablate-6 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-6 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-6 - Train Loss Dict/original_segmentation_loss__MAX","PROPS-NeRF-ablate-5 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-5 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-5 - Train Loss Dict/original_segmentation_loss__MAX","PROPS-NeRF-ablate-4 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-4 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-4 - Train Loss Dict/original_segmentation_loss__MAX","PROPS-NeRF-ablate-3 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-3 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-3 - Train Loss Dict/original_segmentation_loss__MAX","PROPS-NeRF-ablate-2 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-2 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-2 - Train Loss Dict/original_segmentation_loss__MAX","PROPS-NeRF-ablate-1 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-1 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-1 - Train Loss Dict/original_segmentation_loss__MAX","PROPS-NeRF-ablate-0 - Train Loss Dict/original_segmentation_loss","PROPS-NeRF-ablate-0 - Train Loss Dict/original_segmentation_loss__MIN","PROPS-NeRF-ablate-0 - Train Loss Dict/original_segmentation_loss__MAX"
# And plot "segmentation_loss", "rotation_loss", and "translation_loss" where each of these is in their own plot in the same figure, and each plot has 20 lines, one for each ablation model.
# The x-axis is "Step" and the y-axis is the loss value.
# The title of each plot is the name of the loss value.
# The legend of each plot is the name of the ablation model.
# Save the figure as "ablation_training_loss.png".
# Read the file
import csv

# with open('ablation_training_loss.csv', mode='r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         print(row)
#         break

# input_file = csv.DictReader(open("ablation_training_loss.csv"))
# for row in input_file:
#     print(row)
#     break

from pandas import read_csv
# data = read_csv('ablation_training_loss.csv')
# print(data.head())

field_name = lambda idx, field: f'PROPS-NeRF-ablate-{idx} - Train Loss Dict/{field}'

# Create a mathplotlib figure with 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

plot_names = ['segmentation_loss', 'rotation_loss', 'translation_loss']
for idx, plot_name in enumerate(plot_names):
    csv_data = read_csv(f"{plot_name}.csv")
    csv_original_data = read_csv(f"original_{plot_name}.csv")

    data = np.array([csv_data[field_name(i, plot_name)] for i in range(20)])
    original_data = np.array([csv_original_data[field_name(i, f'original_{plot_name}')] for i in range(20)])

    # Get the mean and standard deviation of the data along the first axis
    mean_data = data.mean(axis=0)
    std_data = data.std(axis=0)
    under_data = mean_data - std_data
    over_data = mean_data + std_data

    mean_original_data = original_data.mean(axis=0)
    std_original_data = original_data.std(axis=0)
    under_original_data = mean_original_data - std_original_data
    over_original_data = mean_original_data + std_original_data

    axs[idx].plot(csv_data['Step'], mean_data, label='Trained')
    axs[idx].fill_between(csv_data['Step'], under_data, over_data, alpha=0.2)
    axs[idx].plot(csv_original_data['Step'], original_data.mean(axis=0), label='Original')
    axs[idx].fill_between(csv_data['Step'], under_original_data, over_original_data, alpha=0.2)
    axs[idx].set_title(plot_name)
    axs[idx].legend()

plt.show()





# oseg_data = np.array([data[oseg_name(i)] for i in range(20)])
# seg_data = [data[seg_name(i)] for i in range(20)]
# orot_data = [data[rot_name(i)] for i in range(20)]
# rot_data = [data[rot_name(i)] for i in range(20)]
# otrans_data = [data[otrans_name(i)] for i in range(20)]
# trans_data = [data[trans_name(i)] for i in range(20)]



    

# plt.show()


# Get the columns
# steps = data[:, 0]
# segmentation_loss = data[:, 1:4]
# rotation_loss = data[:, 4:7]
# translation_loss = data[:, 7:10]

# print(steps)
# print(segmentation_loss.shape)

# # Create the figure and subplots
# fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# # Plot segmentation loss
# for i in range(segmentation_loss.shape[1]):
#     axs[0].plot(steps, segmentation_loss[:, i])
# axs[0].set_title('Segmentation Loss')
# axs[0].legend(['Model ' + str(i) for i in range(segmentation_loss.shape[1])])

# # Plot rotation loss
# for i in range(rotation_loss.shape[1]):
#     axs[1].plot(steps, rotation_loss[:, i])
# axs[1].set_title('Rotation Loss')
# axs[1].legend(['Model ' + str(i) for i in range(rotation_loss.shape[1])])

# # Plot translation loss
# for i in range(translation_loss.shape[1]):
#     axs[2].plot(steps, translation_loss[:, i])
# axs[2].set_title('Translation Loss')
# axs[2].legend(['Model ' + str(i) for i in range(translation_loss.shape[1])])

# plt.show()

# # Set common x-axis label
# fig.text(0.5, 0.04, 'Step', ha='center')

# # Save the figure
# plt.savefig('ablation_training_loss.png')