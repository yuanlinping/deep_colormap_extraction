from train import *
import os
import torch
from helper import whatever_to_rgb_numpy
import matplotlib
import re
import shutil
import imageio

matplotlib.use('Agg')
from matplotlib import pyplot as plt

INFER_FILE_LIST = './dataset/test.txt'
RESULT_ROOT = "./experiment_results_and_records/infer_results/"
INFER_DIR = "pred_gt/"
INFER_PRED_DIR = "pred/"
INFER_GT_DIR = "gt/"
ORI_CHART_DIR = "dataset/original_chart/"
model_path = os.path.join(model_dir, config)

def main():
    save_folder_name = PREFIX + "_" + config
    if not os.path.exists(os.path.join(RESULT_ROOT, save_folder_name, INFER_DIR)):
        os.makedirs(os.path.join(RESULT_ROOT, save_folder_name, INFER_DIR))
    else:
        shutil.rmtree(os.path.join(RESULT_ROOT, save_folder_name, INFER_DIR))
        os.makedirs(os.path.join(RESULT_ROOT, save_folder_name, INFER_DIR))

    if not os.path.exists(os.path.join(RESULT_ROOT, save_folder_name, INFER_PRED_DIR)):
        os.makedirs(os.path.join(RESULT_ROOT, save_folder_name, INFER_PRED_DIR))
    else:
        shutil.rmtree(os.path.join(RESULT_ROOT, save_folder_name, INFER_PRED_DIR))
        os.makedirs(os.path.join(RESULT_ROOT, save_folder_name, INFER_PRED_DIR))

    if not os.path.exists(os.path.join(RESULT_ROOT, save_folder_name, INFER_GT_DIR)):
        os.makedirs(os.path.join(RESULT_ROOT, save_folder_name, INFER_GT_DIR))
    else:
        shutil.rmtree(os.path.join(RESULT_ROOT, save_folder_name, INFER_GT_DIR))
        os.makedirs(os.path.join(RESULT_ROOT, save_folder_name, INFER_GT_DIR))


    paths = open(INFER_FILE_LIST, 'r').read().splitlines()
    im_paths = [p.split('\t')[0] for p in paths]  # image
    save_paths = [p.split('/')[-1] for p in im_paths]

    vis_paths = [os.path.join(RESULT_ROOT, save_folder_name, INFER_DIR, "C{}.png".format(re.findall(r"\d+", p)[0])) for p in save_paths]
    vis_pred_paths = [os.path.join(RESULT_ROOT, save_folder_name, INFER_PRED_DIR,"C{}.png".format(re.findall(r"\d+", p)[0])) for p in save_paths]
    vis_gt_paths = [os.path.join(RESULT_ROOT, save_folder_name, INFER_GT_DIR, "C{}.png".format(re.findall(r"\d+", p)[0])) for p in save_paths]

    checkpoint = torch.load(model_path)

    trained_model = VGGModel(input_channel=IMAGE_CHANNEL, label_height=LABEL_HEIGHT, label_width=LABEL_WIDTH)
    if MODE == 2 or MODE == 1:
        if BACKBONE == "vgg":
            trained_model = VGGModel_2D(input_channel=IMAGE_CHANNEL, label_height=LABEL_HEIGHT, label_width=LABEL_WIDTH,
                              with_aspp=WITH_ASPP)
        elif BACKBONE == "resnet18":
            print("resnet18")
            trained_model = ResNet18_2D(input_channel=IMAGE_CHANNEL, label_height=LABEL_HEIGHT, label_width=LABEL_WIDTH,
                              with_aspp=WITH_ASPP)


    trained_model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        trained_model.cuda()
    trained_model.eval()

    test_set = CSV_PNG_Dataset(
        label_paras={'width': LABEL_WIDTH, 'height': LABEL_HEIGHT, 'channel': LABEL_CHANNEL},
        image_paras={'width': IMAGE_WIDTH, 'height': IMAGE_HEIGHT, 'channel': IMAGE_CHANNEL},
        file_list=INFER_FILE_LIST,
        is_label_normalized=IS_LABEL_NORMALIZED
    )

    if MODE == 1:
        test_set = CSV_PNG_Dataset_2D(
            file_list=INFER_FILE_LIST,
            image_paras={'width': IMAGE_WIDTH, 'height': IMAGE_HEIGHT, 'channel': IMAGE_CHANNEL},
            label_paras={'width': LABEL_WIDTH, 'height': LABEL_HEIGHT, 'channel': LABEL_CHANNEL},
            color_space=COLOR_SPACE)
    elif MODE == 2:
        test_set = PNG_PNG_Dataset(label_paras={'width': LABEL_WIDTH, 'height': LABEL_HEIGHT, 'channel': LABEL_CHANNEL}, file_list=INFER_FILE_LIST,
                    color_space=COLOR_SPACE, is_label_normalized=IS_LABEL_NORMALIZED)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    if LOSS_FUNCTION == "MSE":
        criterian = nn.MSELoss()
    elif LOSS_FUNCTION == "BCE":
        criterian = nn.BCELoss()

    loss_record = np.zeros((2,len(vis_paths)))
    for iter, batch in enumerate(test_loader):
        images, labels = batch['image'], batch['label']
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        preds = trained_model(images)
        if IS_LABEL_NORMALIZED:
            preds = sigmoid(preds)

        if LOSS_FUNCTION == "MSE":
            loss = criterian(labels, preds)
        elif LOSS_FUNCTION == "BCE":
            loss = criterian(preds, labels.detach())

        num = re.findall(r'\d+', vis_paths[iter].split('/')[-1])
        loss_record[0][iter] = num[0]
        loss_record[1][iter] = loss

        preds_rgb = whatever_to_rgb_numpy(preds, COLOR_SPACE)
        labels_rgb = whatever_to_rgb_numpy(labels, COLOR_SPACE)

        original_chart = imageio.imread(os.path.join(ORI_CHART_DIR, "C"+num[0]+".png"))

        plt.clf()
        plt.subplot(311)
        plt.imshow(original_chart)
        plt.axis('off')
        plt.title('original chart:{}'.format(num))
        plt.subplot(312)
        plt.imshow(preds_rgb)
        plt.axis('off')
        plt.title('predict+loss:{}'.format(loss))
        plt.subplot(313)
        plt.imshow(labels_rgb)
        plt.axis('off')
        plt.title('ground truth')

        print('Saving to {}'.format(vis_paths[iter]))
        plt.savefig(vis_paths[iter])

        imageio.imwrite(vis_pred_paths[iter],preds_rgb)
        imageio.imwrite(vis_gt_paths[iter], labels_rgb)

        np.savetxt('./experiment_results_and_records/infer_pred_loss/{}.csv'.format(save_folder_name),  loss_record, delimiter=",", fmt="%.7f")

if __name__=="__main__":
    main()
