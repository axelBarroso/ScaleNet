import torch

def save_to_file(info_log, log_writer):
    '''
    Save the training info into a txt file
        :param info_log: str to save into file
        :param log_writer: txt file reference
    '''
    print(info_log)
    file_object = open(log_writer, 'a')
    file_object.write(info_log)
    file_object.close()


def load_ckp(checkpoint_fpath, model, optimizer, device, strict=True):
    '''
        Loads model and optimizer from checkpoint
        :return: loaded model, optimiser and epoch number
    '''
    checkpoint = torch.load(checkpoint_fpath, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=strict)
    if strict:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        epoch = checkpoint['epoch']

    return model, optimizer, epoch