from datetime import datetime
import visdom
from helper import whatever_to_rgb

class Visualizations:
    def __init__(self, env=None):
        if env is None:
            env = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env = env
        self.vis = visdom.Visdom(env=self.env)
        self.loss_win = None

    def plot_loss(self, loss, step, win="loss"):
        self.loss_win = self.vis.line(
            Y=[loss],
            X=[step],
            win="loss",
            update='append' if self.loss_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Loss',
                title='Loss (mean per 10 steps)',
            )
        )

    def plot_test_input(self, images, color_space, win="test_input_image", caption=""):
        image = whatever_to_rgb(images, color_space)
        self.vis.images(image,
            win=win,
            nrow=1,
            opts=dict(
            title='input images',
            caption=caption
            )
    )

    def plot_ground_truth(self, images, color_space, win="test_ground_truth", caption="test_ground_truth"):
        image = whatever_to_rgb(images, color_space)
        self.vis.images(image,win=win, nrow=1, opts=dict(title=caption))

    def plot_test_pred(self, images, color_space, win="test_prediction", caption='test_prediction'):
        image = whatever_to_rgb(images, color_space)
        self.vis.images(image, win=win, nrow=1, opts=dict(title=caption))

    def save(self):
        self.vis.save([self.env])