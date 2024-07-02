import glob
from utils import read_audio, read_from_audio_list
import hyperpyyaml
from tqdm import tqdm
from process import model_infer, metric_evaluation, calc_nsdr
import argparse
import torch
import numpy as np
import os
from utils import get_device
from quantization.qat.models.load_model import create_pretrained_model, enable_observer
import musdb #ubuntu: sudo apt-get install ffmpeg
import museval
DEVICE = get_device()


def argument_handler():
    parser = argparse.ArgumentParser()
    #####################################################################
    # General Config
    #####################################################################
    parser.add_argument('--yml_path', '-y', type=str, required=True, help='YML configuration file')
    parser.add_argument('--use_cpu', action="store_true", help='Use cpu')
    args = parser.parse_args()
    return args


def read_librimix(folder, n_spks=1, noisy=False):
    assert 1<=n_spks<=3, "Error: Up to 3 sources to seperate!"
    if n_spks==1:
        mix_audio_files = sorted(glob.glob(os.path.join(folder, 'mix_single', '*')))
        clean_audio_files = sorted(glob.glob(os.path.join(folder, 's1', '*')))
        assert len(mix_audio_files) == len(clean_audio_files)\
               and len(mix_audio_files) > 0, "Dataset is missing files!"
        return mix_audio_files, [clean_audio_files]
    elif n_spks==2:
        if not noisy:
            mix_audio_files = sorted(glob.glob(os.path.join(folder, 'mix_clean', '*')))
        else:
            mix_audio_files = sorted(glob.glob(os.path.join(folder, 'mix_both', '*')))
        clean1_audio_files = sorted(glob.glob(os.path.join(folder, 's1', '*')))
        clean2_audio_files = sorted(glob.glob(os.path.join(folder, 's2', '*')))
        assert len(mix_audio_files) == len(clean1_audio_files) == len(clean2_audio_files)\
               and len(mix_audio_files) > 0, "Dataset is missing files!"
        return mix_audio_files, [clean1_audio_files, clean2_audio_files]
    elif n_spks==3:
        if not noisy:
            mix_audio_files = sorted(glob.glob(os.path.join(folder, 'mix_clean', '*')))
        else:
            mix_audio_files = sorted(glob.glob(os.path.join(folder, 'mix_both', '*')))
        clean1_audio_files = sorted(glob.glob(os.path.join(folder, 's1', '*')))
        clean2_audio_files = sorted(glob.glob(os.path.join(folder, 's2', '*')))
        clean3_audio_files = sorted(glob.glob(os.path.join(folder, 's3', '*')))
        assert len(mix_audio_files) == len(clean1_audio_files) == len(clean2_audio_files) == len(clean3_audio_files)\
               and len(mix_audio_files) > 0, "Dataset is missing files!"
        return mix_audio_files, [clean1_audio_files, clean2_audio_files, clean3_audio_files]


def val_librimix(model, model_cfg, dataset_cfg, testing_cfg, device):
    # ------------------------------------
    # Read dataset
    # ------------------------------------
    n_srcs = model_cfg.get('n_src', 1)
    mix_audio_files, clean_audio_files_list = read_librimix(testing_cfg['test_dir'], n_srcs, dataset_cfg['noisy'])
    dataset_size = len(mix_audio_files)

    # ------------------------------------
    # Run validation
    # ------------------------------------
    sisdrs, sdrs, stois = np.zeros(dataset_size), np.zeros(dataset_size), np.zeros(dataset_size)
    sisdrs_imp = np.zeros(dataset_size)
    torch.no_grad().__enter__()
    for i in tqdm(range(dataset_size)):
        # Read noisy and clean audios
        mix_wav, fs = read_audio(mix_audio_files[i], resample=dataset_cfg.get('resample',1))
        clean_wavs, _ = read_from_audio_list(clean_audio_files_list, i, resample=dataset_cfg.get('resample',1))
        # Run model
        wavs = model_infer(model,
                           mix_wav,
                           n_srcs=n_srcs,
                           segment=testing_cfg.get('segment_samples', None),
                           overlap=testing_cfg.get('overlap', 0.25),
                           device=device,
                           target=clean_wavs)
        # Metric evaluation
        sisdrs[i], sdrs[i], stois[i] = metric_evaluation(wavs, clean_wavs, sample_rate=fs)
        sisnr_bl, sdr_bl, stoi_bl = metric_evaluation(clean_wavs.squeeze(1), torch.stack([mix_wav]*n_srcs), sample_rate=fs)
        sisdrs_imp[i] = sisdrs[i] - sisnr_bl # SI-SDR improvement
        if i % 500 == 0 and i > 0 or i==1:
            print("SI-SDR={:0.3f},SI-SDR-imp={:0.3f},SDR={:0.3f},STOI={:0.4f}".format(np.mean(sisdrs[:i]),np.mean(sisdrs_imp[:i]),np.mean(sdrs[:i]),np.mean(stois[:i])))

    return np.mean(sisdrs), np.mean(sisdrs_imp), np.mean(sdrs), np.mean(stois)


def val_musdbhq_NSDR(model, testing_cfg, device):
    # ------------------------------------
    # Read dataset
    # ------------------------------------
    mus = musdb.DB(root=testing_cfg['test_dir'], subsets=['test'], is_wav=True)
    assert len(mus.tracks) == 50, "Dataset is missing files!"

    # ------------------------------------
    # Run validation
    # ------------------------------------
    num_sources = len(model.sources)
    sdrs = np.zeros((num_sources, len(mus.tracks)))
    for j,track in tqdm(enumerate(mus)):
        mix = torch.from_numpy(track.audio).t().float()
        # normalize
        ref = mix.mean(dim=0)
        mix_mean, mix_std = ref.mean(), ref.std()
        mix = (mix - mix_mean) / mix_std
        # Run model
        separations = model_infer(model,
                                  mix,
                                  segment=testing_cfg.get('segment_samples',None),
                                  overlap=testing_cfg.get('overlap',0.25),
                                  device=device)
        # denormalize
        separations = separations * mix_std + mix_mean
        for i,src in enumerate(model.sources):
            ref_audio = torch.from_numpy(track.sources[src].audio.T)
            sep_audio = separations[i]
            sdrs[i,j] = calc_nsdr(ref_audio, sep_audio)
        if j % 10 == 0:
            print("\n****** Track {}/{} ******".format(j+1,len(mus.tracks)))
            for i, src in enumerate(model.sources):
                print("{}: SDR={:0.3f}".format(src,sdrs[i,j]))

    sdrs = np.mean(sdrs, axis=1)
    return np.mean(sdrs), sdrs[0], sdrs[1], sdrs[2], sdrs[3]


def val_musdbhq(model, testing_cfg, device):
    # ------------------------------------
    # Read dataset
    # ------------------------------------
    mus = musdb.DB(root=testing_cfg['test_dir'], subsets=['test'], is_wav=True)
    assert len(mus.tracks) == 50, "Dataset is missing files!"

    # ------------------------------------
    # Run validation
    # ------------------------------------
    eval_store = museval.EvalStore()
    signals = model.sources
    track_num = 0
    for track in tqdm(mus):
        mix = torch.from_numpy(track.audio).t().float()
        # normalize
        ref = mix.mean(dim=0)
        mix_mean, mix_std = ref.mean(), ref.std()
        mix = (mix - mix_mean) / mix_std
        # Run model
        separations = model_infer(model,
                                  mix,
                                  segment=testing_cfg.get('segment_samples',None),
                                  overlap=testing_cfg.get('overlap',0.25),
                                  device=device)
        # denormalize
        separations = separations * mix_std + mix_mean
        separations = separations.transpose(2,1).numpy()
        sep_track = {}
        for i,signal in enumerate(signals):
            sep_track.update({signal: separations[i]})
        track_metrics = museval.eval_mus_track(track, sep_track)
        eval_store.add_track(track_metrics)
        if track_num % 10 == 0:
            print(eval_store)
        track_num += 1

    # ------------------------- #
    # Result
    # ------------------------- #
    scores = eval_store.agg_frames_tracks_scores()
    num_signals = len(signals)
    sdrs = np.zeros(num_signals)
    for i,signal in enumerate(signals):
        sdrs[i] = scores[signal]['SDR']

    # Average
    return np.mean(sdrs), sdrs[0], sdrs[1], sdrs[2], sdrs[3]


def val():

    # ------------------------------------
    # Read args
    # ------------------------------------
    args = argument_handler()
    device = "cpu" if args.use_cpu or not torch.cuda.is_available() else 'cuda'
    # Read yml
    with open(args.yml_path) as f:
        conf = hyperpyyaml.load_hyperpyyaml(f)

    # ------------------------------------
    # Load model
    # ------------------------------------
    model_cfg = conf['model_cfg']
    model = create_pretrained_model(model_cfg)
    enable_observer(model, False)
    model.to(device)
    model.eval()

    # ------------------------------------
    # Sanity check
    # ------------------------------------
    assert not (not model_cfg['quantization'].get('qat', False) and (model.n_splitter>1 or model.n_splitter>1)),\
        "No support for splitter/combiner with non QAT model."


    # ------------------------------------
    # Validation
    # ------------------------------------
    dataset_cfg, testing_cfg = conf['dataset_cfg'], conf['testing_cfg']
    if dataset_cfg['name'] == "librimix":
        sisnr, sisnr_imp, sdr, stoi = val_librimix(model, model_cfg, dataset_cfg, testing_cfg, device)
        print("SI-SDR={:0.2f},SI-SDR-imp={:0.2f},SDR={:0.2f},STOI={:0.3f}".format(sisnr, sisnr_imp, sdr, stoi))
    elif dataset_cfg['name'] == "musdbhq":
        if testing_cfg.get("NSDR",False):
            nsdr, nsdr_drums, nsdr_bass, nsdr_other, nsdr_vocals = val_musdbhq_NSDR(model, testing_cfg, device)
            print("NSDR={:0.2f},NSDR_DRUMS={:0.2f},NSDR_BASS={:0.2f},NSDR_OTHER={:0.2f},SNDR_VOCALS={:0.2f}".format(nsdr,nsdr_drums,nsdr_bass,nsdr_other,nsdr_vocals))
        else:
            sdr, sdr_drums, sdr_bass, sdr_other, sdr_vocals = val_musdbhq(model, testing_cfg, device)
            print("SDR={:0.2f},SDR_DRUMS={:0.2f},SDR_BASS={:0.2f},SDR_OTHER={:0.2f},SDR_VOCALS={:0.2f}".format(sdr,sdr_drums,sdr_bass,sdr_other,sdr_vocals))
    else:
        assert False, "Dataset {} is not supported!".format(dataset_cfg['name'])


if __name__ == '__main__':
    val()