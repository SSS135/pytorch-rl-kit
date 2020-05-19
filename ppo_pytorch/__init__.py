if __name__ == '__main__':
    import torch
    nt = 8 if __name__ == '__main__' else 1
    torch.set_num_threads(nt)
    torch.set_num_interop_threads(nt)

# from . import common
# from . import actors
# from . import algs

from .common.procgen_easy import register_environments
register_environments()