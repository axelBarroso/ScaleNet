import torch
from tqdm import tqdm
import torch.nn.functional as F

def criterion_KL_divergence(dist1, dist2, beta=100):
    """
    KL divergence loss function
    Args:
        dist1: scale distribution predicted by ScaleNet
        dist2: gt scale distribution
    Output:
        loss: KL-D loss value on the batch
    """

    loss = beta*F.kl_div(F.log_softmax(dist1, 1), dist2, reduction="none").mean()

    return loss


def train_epoch(net, optimizer, train_loader, scale_distr, device):
    """
    Training epoch script
    Args:
        net: model architecture
        optimizer: optimizer to be used for traninig `net`
        train_loader: dataloader
        scale_distr: scale classes in the distribution
        device: `cpu` or `gpu`
    Output:
        running_total_loss: total training loss
        running_total_scale_diff: scale factor difference between predictions y gt
    """

    net.train()
    running_total_loss = 0
    running_diff_scale = 0
    softmax = torch.nn.Softmax(1)
    scale_distr = torch.from_numpy(scale_distr).to(device).float()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for i, mini_batch in pbar:

        scale = net(mini_batch['source_image'].to(device), mini_batch['target_image'].to(device))
        scale_gt = mini_batch['scale'].to(device).float()

        scale_value = torch.exp(torch.sum(scale_distr * softmax(scale), 1))
        scale_value_gt = mini_batch['scale_factor'].to(device).float()

        loss_net = criterion_KL_divergence(scale, scale_gt)

        max_scale = torch.where(scale_value > scale_value_gt, scale_value, scale_value_gt)
        min_scale = torch.where(scale_value < scale_value_gt, scale_value, scale_value_gt)

        diff_scale = torch.exp(torch.log(max_scale / min_scale).mean()).item()

        loss_net.backward()
        optimizer.step()
        running_total_loss += loss_net.item()
        running_diff_scale += diff_scale
        pbar.set_description('R_total_loss: %.3f/%.3f. Diff Scale: %.3f/%.3f' %
                             (running_total_loss / (i + 1), loss_net.item(), running_diff_scale / (i + 1), diff_scale))

    running_total_loss /= len(train_loader)
    return running_total_loss, running_diff_scale / len(train_loader)


def validate_epoch(net, val_loader, scale_distr, device):
    """
    Validation epoch script
    Args:
        net: model architecture
        val_loader: dataloader
        scale_distr: scale classes in the distribution
        device: `cpu` or `gpu`
    Output:
        running_total_loss: total training loss
        running_total_scale_diff: scale factor difference between predictions y gt
    """

    net.eval()
    softmax = torch.nn.Softmax(1)
    scale_distr = torch.from_numpy(scale_distr).to(device).float()
    running_total_loss = 0
    running_diff_scale = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, mini_batch in pbar:

            # net predictions
            scale = net(mini_batch['source_image'].to(device), mini_batch['target_image'].to(device))
            scale_gt = mini_batch['scale'].to(device).float()
            scale_value_gt = mini_batch['scale_factor'].to(device).float()
            scale_value = torch.exp(torch.sum(scale_distr * softmax(scale), 1))
            loss_net = criterion_KL_divergence(scale, scale_gt)

            max_scale = torch.where(scale_value > scale_value_gt, scale_value, scale_value_gt)
            min_scale = torch.where(scale_value < scale_value_gt, scale_value, scale_value_gt)

            diff_scale = torch.exp(torch.log(max_scale / min_scale).mean()).item()

            running_total_loss += loss_net.item()
            running_diff_scale += diff_scale
            pbar.set_description('R_total_loss: %.3f/%.3f. Diff Scale: %.3f/%.3f' %
                                 (running_total_loss / (i + 1), loss_net.item(),
                                  running_diff_scale / (i + 1), diff_scale))

    return running_total_loss / len(val_loader), running_diff_scale / len(val_loader)