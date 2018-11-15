# -*- coding: utf-8 -*-
"""
author:  LeeJiangLee
contact: ljllili23@gmail.com

8/11/2018 12:29 PM
"""



class mnistDataset(Dataset):
    """mnist dataset."""
    def __init__(self,csv_file,transform=None):
        self.mnist_frame = pd.read_csv(csv_file)

        self.transform = transform

    def __len__(self):
        return len(self.mnist_frame)

    def __getitem__(self,idx):
        mnist_data = self.mnist_frame.iloc[idx,1:].values
        # mnist_data = mnist_data.astype('float').reshape(28,28)
        mnist_data = mnist_data.astype('float')
        mnist_label = self.mnist_frame.iloc[idx,0]
        # sample = {'image':mnist_data, 'label':mnist_label}

        # if self.transform:
        #     sample = self.transform(sample)

        return mnist_data, mnist_label

# mnist_dataset = mnistDataset(csv_file='MNIST/train.csv')
#
# for i in range(len(mnist_dataset)):
#     sample = mnist_dataset[i]
#     print(i,sample['data'].shape, sample['label'].dtype)

# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors"""
#     def __call__(self,sample):
#         image, label = sample['image'],sample['label']
#         return {'image':torch.from_numpy(image),
#                 'label':torch.unsqueeze(label,1)}

# transformed_dataset = mnistDataset(
#     csv_file='./MNIST/train.csv',
#     transform=transforms.Compose([ToTensor()])
# )
transformed_dataset = mnistDataset(
    csv_file='./train.csv',

)
train_size = int(0.8*len(transformed_dataset))
test_size = len(transformed_dataset)-train_size
train_dataset,test_dataset = random_split(transformed_dataset,[train_size,test_size])
# for i in range(len(transformed_dataset)):
#     sample = transformed_dataset[i]
#     print(i,sample['image'].size(),sample['label'])
#
#     if i==3:
#         break

#Batching the data
#Shuffling the data
#Load the data in parallel using multiprocessing workers
train_dataloader = DataLoader(train_dataset,batch_size=20,shuffle=True)

# # Helper function to show a batch
# def show_mnist_batch(sample_batched):
#     images_batch,labels_batch = \
#     sample_batched['image'],sample_batched['label']
#     batch_size = len(images_batch)
#     grid = utils.make_grid(images_batch)

# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch,sample_batched['image'].size())
#     if i_batch == 3:
#         break

model = torch.nn.Sequential(
    torch.nn.Linear(784,1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024,10)
)

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-4
for i_batch, (data, label) in enumerate(train_dataloader):
    y_pred = model(data)
    loss = loss_fn(y_pred,label)
    print(i_batch,loss.item())

    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate*param.grad

now = datetime.datetime.now()
current_time = str(now.date())+'-'+str(now.hour)+'-'+str(now.minute)
torch.save(model,'./model/'+current_time)