from torch.utils.data.dataset import Dataset

class BaseDataset(Dataset):
   def __init__(self,file_list="./dataset/train.txt",
                image_paras = {'width':64,'height':64,'channel':64},
                label_paras ={'width':256,'height':10,'channel':3}):
      super(BaseDataset, self).__init__()

      #constant value
      self.IMAGE_WIDTH = image_paras['width']
      self.IMAGE_HEIGHT = image_paras['height']
      self.IMAGE_CHANNEL = image_paras['channel']

      self.LABEL_WIDTH = label_paras['width']
      self.LABEL_HEIGHT = label_paras['height']
      self.LABEL_CHANNEL = label_paras['channel']

      paths = open(file_list, 'r').read().splitlines()
      self.image_paths = [p.split('\t')[0] for p in paths]
      self.label_paths = [p.split('\t')[1] for p in paths]


   def __len__(self):
      return len(self.image_paths)

   def __getitem__(self, idx):
      raise NotImplementedError


if __name__=='__main__':
   dataset = BaseDataset()
   print(dataset.image_paths)
   print(dataset.label_paths)
   print(len(dataset.image_paths))