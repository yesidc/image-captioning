import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
from stats.swin_F_1_GPT_2_training_loss import training

data = {'loss':[], 'learning_rate':[], 'epoch':[]}

for i in training:
    data['loss'].append(i['loss'])
    data['learning_rate'].append(i['learning_rate'])
    data['epoch'].append(i['epoch'])





# Create a line plot of loss and learning rate
plt.plot(data['epoch'], data['loss'], label='Loss')
plt.plot(data['epoch'], data['learning_rate'], label='Learning Rate')

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Loss and Learning Rate over Epochs')

# Add legend
plt.legend()

# Show the plot
plt.show()

print()
