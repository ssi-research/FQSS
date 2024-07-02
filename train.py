import argparse
import torch
import sys
import train_env.asteroid_librimix.asteroid_librimix_trainer as asteroid_librimix_trainer
import train_env.tasnet_musdbhq.tasnet_musdbhq_trainer as tasnet_musdbhq_trainer
import train_env.htdemucs_musdbhq.train as htdemucs_musdbhq_trainer
import train_env.speechbrain_librimix.speechbrain_librimix_trainer as speechbrain_librimix_trainer


def argument_handler():
    parser = argparse.ArgumentParser()
    #####################################################################
    # General Config
    #####################################################################
    parser.add_argument('--env_name', '-env', type=str, required=True, help='Training environment name: asteroid/tasnet/speechbrain/htdemucs')
    parser.add_argument('--yml_path', '-y', type=str, required=True, help='YML configuration file')
    parser.add_argument('--use_cpu', action="store_true", help='Use cpu')
    parser.add_argument('--local_rank', type=int, default=0, help='Rank ID')
    parser.add_argument('--distributed_launch', action="store_true", help='Multi-GPU training')

    args = parser.parse_args()
    return args


def train():

    # ------------------------------------
    # Read args
    # ------------------------------------
    args = argument_handler()
    device = "cpu" if args.use_cpu or not torch.cuda.is_available() else 'cuda'

    # ------------------------------------
    # Run training
    # ------------------------------------
    torch.cuda.empty_cache()
    if args.env_name == "asteroid":
        asteroid_librimix_trainer.train(args.yml_path, device)
    elif args.env_name == "speechbrain":
        speechbrain_librimix_trainer.train(args.yml_path, args.local_rank, args.distributed_launch, device)
    elif args.env_name == "tasnet":
        tasnet_musdbhq_trainer.train(args.yml_path, device)
    elif args.env_name == "htdemucs":
        sys.argv[1:] = []
        sys.argv.append("+device="+device)
        htdemucs_musdbhq_trainer.main()
    else:
        assert False, "Training environment {} is not supported!".format(args.env_name)

    print("Training is done!")

if __name__ == '__main__':
    train()















