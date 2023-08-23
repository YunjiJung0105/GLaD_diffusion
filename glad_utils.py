import torch
import numpy as np
import copy
import utils
import wandb
import os, sys
import torchvision
import torchvision.transforms as T
import gc
from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image
from natsort import natsorted
import csv

from utils import get_network, config, evaluate_synset, TensorDataset
sys.path.append(os.getcwd())
from latent_diffusion.ldm.util import instantiate_from_config
from latent_diffusion.ldm.models.diffusion.ddim import DDIMSampler



def build_dataset(ds, class_map, num_classes):
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]    # indices per class
    print("BUILDING DATASET")
    for i in tqdm(range(len(ds))):
        sample = ds[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])
    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")
    print(images_all.shape)   # CIFAR 10 / 100 = [50000, 3, 32, 32]
    print(labels_all)   # CIFAR 10 / 100 = [50000]
    print(len(indices_class))   # CIFAR 10 = [10] / CIFAR 100 = [100]
    
    return images_all, labels_all, indices_class


def prepare_latents(channel=4, num_classes=10, im_size=(256, 256), embed_dim=512, G=None, class_map_inv={}, get_images=None, args=None):
    with torch.no_grad():
        # label
        label_syn = torch.tensor([i*np.ones(args.ipc, dtype=np.int64) for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        # latents
        latents = torch.randn(size=(num_classes * args.ipc, channel, 64, 64), dtype=torch.float, requires_grad=True, device=args.device)
        # latents = latents.detach().to(args.device).requires_grad_(True)
            
        # cond_emb
        cond_emb = torch.randn(label_syn.shape[0], 1, embed_dim, dtype=torch.float, requires_grad=True, device=args.device)

        return latents, cond_emb, label_syn


def get_optimizer_img(latents=None, cond_emb=None, G=None, args=None):
    optimizer_img = torch.optim.SGD([cond_emb], lr=args.lr_w, momentum=0.5)
    optimizer_img.add_param_group({'params': latents, 'lr': args.lr_img, 'momentum': 0.5})

    if args.learn_g:
        G.requires_grad_(True)
        optimizer_img.add_param_group({'params': G.parameters(), 'lr': args.lr_g, 'momentum': 0.5})

    optimizer_img.zero_grad()

    return optimizer_img

def get_eval_lrs(args):
    eval_pool_dict = {
        args.model: 0.001,
        "ResNet18": 0.001,
        "VGG11": 0.0001,
        "AlexNet": 0.001,
        "ViT": 0.001,

        "AlexNetCIFAR": 0.001,
        "ResNet18CIFAR": 0.001,
        "VGG11CIFAR": 0.0001,
        "ViTCIFAR": 0.001,
    }

    return eval_pool_dict


def eval_loop(latents=None, cond_emb=None, label_syn=None, G=None, best_acc={}, best_std={}, testloader=None, model_eval_pool=[], it=0, channel=3, num_classes=10, im_size=(32, 32), args=None):
    curr_acc_dict = {}
    max_acc_dict = {}

    curr_std_dict = {}
    max_std_dict = {}

    eval_pool_dict = get_eval_lrs(args)

    save_this_it = False
    
    for model_eval in model_eval_pool:

        if model_eval != args.model and args.wait_eval and it != args.Iteration:
            continue
        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (
        args.model, model_eval, it))

        accs_test = []
        accs_train = []

        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size, width=args.width, depth=args.depth,
                                   dist=False).to(args.device)  # get a random model
            eval_lats = latents
            eval_labs = label_syn
            image_syn = latents
            image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                eval_labs.detach())  # avoid any unaware modification

            with torch.no_grad():
                image_syn_eval = torch.cat(
                    [latent_to_im(G, image_syn_eval_split, cond_emb_split, args=args).detach() for
                        image_syn_eval_split, cond_emb_split, label_syn_split in
                        zip(torch.split(image_syn_eval, args.sg_batch), torch.split(cond_emb, args.sg_batch),
                            torch.split(label_syn, args.sg_batch))])

            args.lr_net = eval_pool_dict[model_eval]
            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader,
                                                     args=args, aug=True)
            del _
            del net_eval
            accs_test.append(acc_test)
            accs_train.append(acc_train)

        print(accs_test)
        accs_test = np.array(accs_test)
        accs_train = np.array(accs_train)
        acc_test_mean = np.mean(np.max(accs_test, axis=1))
        acc_test_std = np.std(np.max(accs_test, axis=1))
        best_dict_str = "{}".format(model_eval)
        if acc_test_mean > best_acc[best_dict_str]:
            best_acc[best_dict_str] = acc_test_mean
            best_std[best_dict_str] = acc_test_std
            save_this_it = True

        curr_acc_dict[best_dict_str] = acc_test_mean
        curr_std_dict[best_dict_str] = acc_test_std

        max_acc_dict[best_dict_str] = best_acc[best_dict_str]
        max_std_dict[best_dict_str] = best_std[best_dict_str]

        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
        len(accs_test[:, -1]), model_eval, acc_test_mean, np.std(np.max(accs_test, axis=1))))
        wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
        wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[best_dict_str]}, step=it)
        wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
        wandb.log({'Max_Std/{}'.format(model_eval): best_std[best_dict_str]}, step=it)

    wandb.log({
        'Accuracy/Avg_All'.format(model_eval): np.mean(np.array(list(curr_acc_dict.values()))),
        'Std/Avg_All'.format(model_eval): np.mean(np.array(list(curr_std_dict.values()))),
        'Max_Accuracy/Avg_All'.format(model_eval): np.mean(np.array(list(max_acc_dict.values()))),
        'Max_Std/Avg_All'.format(model_eval): np.mean(np.array(list(max_std_dict.values()))),
    }, step=it)

    curr_acc_dict.pop("{}".format(args.model))
    curr_std_dict.pop("{}".format(args.model))
    max_acc_dict.pop("{}".format(args.model))
    max_std_dict.pop("{}".format(args.model))

    wandb.log({
        'Accuracy/Avg_Cross'.format(model_eval): np.mean(np.array(list(curr_acc_dict.values()))),
        'Std/Avg_Cross'.format(model_eval): np.mean(np.array(list(curr_std_dict.values()))),
        'Max_Accuracy/Avg_Cross'.format(model_eval): np.mean(np.array(list(max_acc_dict.values()))),
        'Max_Std/Avg_Cross'.format(model_eval): np.mean(np.array(list(max_std_dict.values()))),
    }, step=it)

    return save_this_it


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("/raid/workspace/cvml_user/jyj/docker_share/latent-diffusion/configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "/raid/workspace/cvml_user/jyj/docker_share/latent-diffusion/models/ldm/cin256-v2/model.ckpt")
    return model


def latent_to_im(G, latents, cond_emb, args=None):    
    ddim_steps = 50
    scale = 3.0
    sampler = DDIMSampler(G)

    with G.ema_scope():
        # uc = G.get_learned_conditioning(
        #     {G.cond_stage_key: torch.tensor(args.sg_batch*[1000]).to(G.device)}
        #     )
        # batch here
        
        print(f"rendering {args.sg_batch} examples in {ddim_steps} steps and using s={scale:.2f}.")            
        samples_ddim, intermediates = sampler.sample(S=ddim_steps,
                                        x_T=latents,
                                        conditioning=cond_emb, 
                                        batch_size=args.sg_batch,
                                        shape=[3,64,64],
                                        verbose=False,
                                        # unconditional_guidance_scale=scale,
                                        # unconditional_conditioning=uc, 
                                        eta=0.0)

        imgs = intermediates['pred_x0'][-1]    
        imgs = G.decode_first_stage(imgs)
        imgs = torch.clamp((imgs+1.0)/2.0, min=0.0, max=1.0)

    return imgs


def image_logging(latents=None, cond_emb=None, label_syn=None, G=None, it=None, save_this_it=None, args=None):
    with torch.no_grad():
        image_syn = latents.cuda()

        with torch.no_grad():
            if args.layer is None or args.layer == -1:
                image_syn = latent_to_im(G, image_syn.detach(), cond_emb.detach(), args=args)
            else:
                image_syn = torch.cat(
                    [latent_to_im(G, image_syn_split.detach(), cond_emb_split.detach(), args=args).detach() for
                        image_syn_split, cond_emb_split, label_syn_split in
                        zip(torch.split(image_syn, args.sg_batch),
                            torch.split(cond_emb, args.sg_batch),
                            torch.split(label_syn, args.sg_batch))])

        save_dir = os.path.join(args.logdir, args.dataset, wandb.run.name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(image_syn.cpu(), os.path.join(save_dir, "images_{0:05d}.pt".format(it)))
        torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_{0:05d}.pt".format(it)))

        if save_this_it:
            torch.save(image_syn.cpu(), os.path.join(save_dir, "images_best.pt".format(it)))
            torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_best.pt".format(it)))

        wandb.log({"Latent_Codes": wandb.Histogram(torch.nan_to_num(latents.detach().cpu()))}, step=it)

        if args.ipc < 50 or args.force_save:

            upsampled = image_syn
            if "imagenet" not in args.dataset:
                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
            wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
            wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)

            for clip_val in []:
                upsampled = torch.clip(image_syn, min=-clip_val, max=clip_val)
                if "imagenet" not in args.dataset:
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                wandb.log({"Clipped_Synthetic_Images/raw_{}".format(clip_val): wandb.Image(
                    torch.nan_to_num(grid.detach().cpu()))}, step=it)

            for clip_val in [2.5]:
                std = torch.std(image_syn)
                mean = torch.mean(image_syn)
                upsampled = torch.clip(image_syn, min=mean - clip_val * std, max=mean + clip_val * std)
                if "imagenet" not in args.dataset:
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(
                    torch.nan_to_num(grid.detach().cpu()))}, step=it)

    del upsampled, grid


def model_backward(latents=None, cond_emb=None, image_syn=None, G=None, args=None):
    cond_emb.grad = None
    latents_grad_list = []
    cond_emb_grad_list = []
    for latents_split, cond_emb_split, dLdx_split in zip(torch.split(latents, args.sg_batch),
                                                          torch.split(cond_emb, args.sg_batch),
                                                          torch.split(image_syn.grad, args.sg_batch)):
        latents_detached = latents_split.detach().clone().requires_grad_(True)
        cond_emb_detached = cond_emb_split.detach().clone().requires_grad_(True)

        syn_images = latent_to_im(G, latents_detached, cond_emb_detached, args=args)
        syn_images.requires_grad_(True)
        syn_images.backward((dLdx_split,))

        latents_grad_list.append(latents_detached.grad)
        cond_emb_grad_list.append(cond_emb_detached.grad)
        print(latents_grad_list)
        print(cond_emb_grad_list)

        del syn_images
        del latents_split
        del cond_emb_split
        del dLdx_split
        del cond_emb_detached
        del latents_detached

        gc.collect()

    latents.grad = torch.cat(latents_grad_list)
    del latents_grad_list
    if args.layer != -1:
        cond_emb.grad = torch.cat(cond_emb_grad_list)
        del cond_emb_grad_list


