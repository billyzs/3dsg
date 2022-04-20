import matplotlib.pyplot as plt
import torch


class TrainingLogger:
    def __init__(self, title, train_n, print_interval):
        self.model_title = title
        file_name = title.lower().replace(" ", "_")
        self.save_path = "{}.pt".format(file_name)
        self.epoch = 0
        self.train_n = train_n

        self.train_losses = []
        self.val_losses = []
        self.val_state_confs = []
        self.val_pos_confs = []
        self.val_mask_confs = []
        self.val_state_accs = []
        self.val_pos_accs = []
        self.val_mask_accs = []

        self.print_interval = print_interval
        self.train_iter_count = 0
        self.curr_train_loss = 0

        self.val_iter_count = 0
        self.curr_val_loss = 0
        self.val_state_conf = torch.zeros((2, 2))
        self.val_pos_conf = torch.zeros((2, 2))
        self.val_mask_conf = torch.zeros((2, 2))

        self.latest_model = None

    def log_training_iter(self, loss):
        self.curr_train_loss += loss
        self.train_iter_count += 1
        if self.train_iter_count % self.print_interval == 0:
            print("{}/{}, loss = {}".format(self.train_iter_count, self.train_n, self.curr_train_loss/self.train_iter_count))

    def log_training_epoch(self, model):
        avg_train_loss = self.curr_train_loss/self.train_iter_count
        if len(self.train_losses) > 0:
            if avg_train_loss < min(self.train_losses):
                print("saving model")
                self.latest_model = model.state_dict()
                self.save_model()
        self.train_losses.append(avg_train_loss)
        self.curr_train_loss = 0
        self.train_iter_count = 0

    def log_validation_iter(self, loss, state_conf, pos_conf, mask_conf):
        self.val_state_conf += state_conf
        self.val_pos_conf += pos_conf
        self.val_mask_conf += mask_conf
        self.curr_val_loss += loss
        self.val_iter_count += 1

    def log_validation_epoch(self):
        self.val_state_confs.append(self.val_state_conf)
        self.val_pos_confs.append(self.val_pos_conf)
        self.val_mask_confs.append(self.val_mask_conf)

        self.val_state_accs.append((self.val_state_conf[0, 0] + self.val_state_conf[1, 1]) / torch.sum(self.val_state_conf))
        self.val_pos_accs.append((self.val_pos_conf[0, 0] + self.val_pos_conf[1, 1]) / torch.sum(self.val_pos_conf))
        self.val_mask_accs.append((self.val_mask_conf[0, 0] + self.val_mask_conf[1, 1]) / torch.sum(self.val_mask_conf))
        self.val_losses.append(self.curr_val_loss/self.val_iter_count)

        self.val_iter_count = 0
        self.curr_val_loss = 0
        self.val_state_conf = torch.zeros((2, 2))
        self.val_pos_conf = torch.zeros((2, 2))
        self.val_mask_conf = torch.zeros((2, 2))

    def plot_training_losses(self):
        plt.plot(self.train_losses)
        plt.ylabel("Training Losses")
        plt.xlabel("Epoch")
        plt.title(self.model_title)
        plt.show()

    def plot_valid_losses(self):
        plt.plot(self.val_losses)
        plt.ylabel("Validation Losses")
        plt.xlabel("Epoch")
        plt.title(self.model_title)
        plt.show()

    def plot_accuracies(self):
        plt.plot(self.val_state_accs, label="State Variability")
        plt.plot(self.val_state_accs, label="Position Variability")
        plt.plot(self.val_state_accs, label="Mask")
        plt.ylabel("Validation Accuracies")
        plt.xlabel("Epoch")
        plt.title(self.model_title)
        plt.legend()
        plt.show()

    def save_model(self):
        torch.save(self.latest_model, self.save_path)


class EvalLogger:
    def __init__(self, title):
        self.model_title = title

        self.state_conf = torch.zeros((2, 2))
        self.pos_conf = torch.zeros((2, 2))
        self.mask_conf = torch.zeros((2, 2))
        self.iter_count = 0

    def log_eval_iter(self, state_conf, pos_conf, mask_conf):
        self.state_conf += state_conf
        self.pos_conf += pos_conf
        self.mask_conf += mask_conf
        self.iter_count += 1

    def print_eval_results(self):
        print("{} Results".format(self.model_title))
        print(" State Variability:")
        print("     Accuracy: {}".format((self.state_conf[0, 0] + self.state_conf[1, 1]) / torch.sum(self.state_conf)))
        print("     Precision: {}".format(self.state_conf[0, 0]/torch.sum(self.state_conf[0, :])))
        print("     Recall: {}".format(self.state_conf[0, 0]/torch.sum(self.state_conf[:, 0])))
        print(" Position Variability:")
        print("     Accuracy: {}".format((self.pos_conf[0, 0] + self.pos_conf[1, 1]) / torch.sum(self.pos_conf)))
        print("     Precision: {}".format(self.pos_conf[0, 0] / torch.sum(self.pos_conf[0, :])))
        print("     Recall: {}".format(self.pos_conf[0, 0] / torch.sum(self.pos_conf[:, 0])))
        print(" Node Variability:")
        print("     Accuracy: {}".format((self.mask_conf[0, 0] + self.mask_conf[1, 1]) / torch.sum(self.mask_conf)))
        print("     Precision: {}".format(self.mask_conf[0, 0] / torch.sum(self.mask_conf[0, :])))
        print("     Recall: {}".format(self.mask_conf[0, 0] / torch.sum(self.mask_conf[:, 0])))