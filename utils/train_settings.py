import os
import shutil



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', *args):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = directory + filename
    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

