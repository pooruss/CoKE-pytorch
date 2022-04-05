import torch
import torch.nn.functional as F
import math
from init_args import create_args

args, logger = create_args()

class Trainer():
    def __init__(self, model, train_config):
        self.model = model
        self.model_name = train_config["model_name"]
        self.do_train = train_config["do_train"]
        self.do_val = train_config["do_val"]
        self.do_test = train_config["do_test"]
        self.batch_size = train_config["batch_size"]
        self.epoch = train_config["epoch"]
        self.use_cuda = train_config["use_cuda"]
        self.checkpoint_num = train_config["checkpoint_num"]
        self.learning_rate = train_config["learning_rate"]
        self.skip_steps = train_config["skip_steps"]
        self.save_path = train_config["save_path"]
        self.log_file = open('./train2.log', 'w', encoding='utf-8')
        self.loss_function = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.nodes = (1 if self.use_cuda == False else train_config["nodes"])
        self.gpus = (1 if self.use_cuda == False else train_config["gpus"])

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def load_train_data_loader(self, train_data_loader):
        self.train_data_loader = train_data_loader

    def load_val_data_loader(self, val_data_loader):
        self.val_data_loader = val_data_loader

    def prediction(self):
        return True

    def validation(self):
        self.model.eval()
        acc, loss_sum = 0.0, 0.0

        step_per_epoch = len(self.val_data_loader)
        print("----------- Evaluation -----------")
        self.log_file.write(str(len(self.val_data_loader))+'\n')
        for iter, batch_data in enumerate(self.val_data_loader):
            # fetch batch data
            try:
                src_id, pos_id, input_mask, mask_pos, mask_label = batch_data
            except RuntimeError:
                print("Per data instance's length should be 5, received {}.".format(len(batch_data)))
                continue
            if self.use_cuda:
                src_id, pos_id, input_mask, mask_pos, mask_label = \
                    src_id.cuda(), pos_id.cuda(), input_mask.cuda(), mask_pos.cuda(), mask_label.cuda()
            input_x = [src_id, pos_id, input_mask, mask_pos]
            # forward
            y_hat = self.model(input_x)
            # loss
            loss = self.loss_function(
                input=y_hat,
                target=mask_label.squeeze()
            )
            acc += y_hat.max(dim=1)[1].eq(mask_label.squeeze()).sum().data
            loss_sum += loss.data
            total_acc = acc * 100 / len(self.val_data_loader.dataset)
            total_loss = loss_sum / len(self.val_data_loader)
            if iter % self.skip_steps == 0:
                print('Step:{}/{}, loss:{}, acc:{}.'.format(
                    str(iter),
                    str(step_per_epoch),
                    str(total_loss.cpu().detach().numpy()), str(total_acc.cpu().detach().numpy())))
                self.log_file.write(('Step:{}/{}, loss:{}, acc:{}.'.format(
                    str(iter),
                    str(step_per_epoch),
                    str(total_loss.cpu().detach().numpy()), str(total_acc.cpu().detach().numpy()))) + '\n')

    def train(self):
        acc, loss_sum = 0.0, 0.0
        step_per_epoch = len(self.train_data_loader)
        total_train_step = step_per_epoch * self.epoch
        save_step = math.ceil(total_train_step / self.nodes / self.gpus / self.checkpoint_num)
        for epoch in range(self.epoch):
            for iter, batch_data in enumerate(self.train_data_loader):
                # fetch batch data
                try:
                    src_id, pos_id, input_mask, mask_pos, mask_label = batch_data
                except RuntimeError:
                    print("One data instance's length should be 5, received {}.".format(len(batch_data)))
                    continue
                if self.use_cuda:
                    src_id, pos_id, input_mask, mask_pos, mask_label = \
                    src_id.cuda(), pos_id.cuda(), input_mask.cuda(), mask_pos.cuda(), mask_label.cuda()
                input_x = {
                    'src_ids':src_id,
                    'position_ids':pos_id,
                    'input_mask':input_mask,
                    'mask_pos':mask_pos,
                    'mask_label':mask_label
                }
                # forward
                logits = self.model(input_x).cuda() if self.use_cuda else self.model(input_x)
                # loss
                loss = F.cross_entropy(
                            input=logits,
                            target=mask_label.squeeze(),
                            label_smoothing=0.1
                        )
                # backward
                if self.do_train:
                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    self.optimizer.step()

                # log
                lr = self.lr_scheduler.get_last_lr()
                acc = logits.max(dim=1)[1].eq(mask_label.squeeze()).sum()
                acc = acc / float(batch_data[0].size(0))
                current_step = epoch * step_per_epoch + iter

                if iter % self.skip_steps == 0 or iter+1 == step_per_epoch:
                    print('Epoch:{}, Step:{}/{}, loss:{}, acc:{}, lr:{}.'.format(
                        str(epoch), str(current_step),
                        str(total_train_step),
                        str(loss.cpu().detach().numpy()),
                        str(acc.cpu().detach().numpy()),
                        str(lr))
                    )
                    self.log_file.write('Epoch:{}, Step:{}/{}, loss:{}, acc:{}, lr{}:.'.format(
                        str(epoch), str(current_step),
                        str(total_train_step),
                        str(loss.cpu().detach().numpy()),
                        str(acc.cpu().detach().numpy()),
                        str(lr))+'\n'
                    )

                if current_step == save_step:
                    if isinstance(self.model, torch.nn.DataParallel):
                        torch.save(self.model.module.state_dict(), self.save_path+'{}_lr{}_bs{}_step{}.pt'.format(
                            self.model_name, self.learning_rate,
                            self.batch_size, str(current_step)))
                    else:
                        torch.save(self.model.state_dict(), self.save_path+'{}_lr{}_bs{}_step{}.pt'.format(
                            self.model_name, self.learning_rate,
                            self.batch_size, str(current_step)))

            self.lr_scheduler.step()

        return acc, loss_sum
